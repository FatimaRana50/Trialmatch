"""Hand-written Qwen2 decoder forward pass with an explicit KV cache.

The Hugging Face model is used only as a weight container: embedding, the q/k/v/o and
MLP linear layers, and the RMSNorm modules. Rotary position embeddings, the KV cache,
grouped-query attention, the causal mask and the residual stream are implemented here.

Cache layout (KVCache): per layer, preallocated K and V buffers [B, n_kv_heads, max_len, head_dim].
`length` = number of positions already written. A forward call with T new tokens
  1. gives them positions length .. length+T-1 (RoPE uses these absolute positions),
  2. writes their K/V into slots length .. length+T-1 of every layer,
  3. lets each new query attend to keys 0 .. its own position (offset causal mask),
  4. advances length by T.
`crop(n)` rolls the cache back to n positions (used by speculative decoding on rejection);
stale slots after n are simply overwritten later.
"""
import torch

from attention import ATTENTION


class KVCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, dtype, device):
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = max_len
        self.length = 0

    def write(self, layer, k_new, v_new, start):
        end = start + k_new.shape[-2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} slots, have {self.max_len}")
        self.k[layer][:, :, start:end] = k_new
        self.v[layer][:, :, start:end] = v_new
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def crop(self, n):
        assert 0 <= n <= self.length
        self.length = n

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ManualQwen2:
    def __init__(self, hf_model, attn="sdpa"):
        cfg = hf_model.config
        self.hf = hf_model
        self.embed = hf_model.model.embed_tokens
        self.layers = hf_model.model.layers
        self.final_norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # transformers 5 keeps RoPE settings in rope_parameters; 4.x used rope_theta/rope_scaling
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        theta = getattr(cfg, "rope_theta", None) or rope.get("rope_theta")
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise NotImplementedError("only plain RoPE (no scaling) is implemented")
        dev = hf_model.device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=dev).float() / self.head_dim))
        self.dtype = next(hf_model.parameters()).dtype
        self.device = dev
        self.attn_name = attn

    def new_cache(self, max_len, batch=1):
        return KVCache(len(self.layers), batch, self.n_kv, max_len, self.head_dim, self.dtype, self.device)

    def rope(self, positions):
        """cos/sin tables for absolute positions, [T, head_dim] each (same as HF Qwen2)."""
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.no_grad()
    def forward(self, input_ids, cache=None, all_logits=False):
        """input_ids [B, T]. With a cache, the tokens continue from cache.length.
        Without a cache, input_ids must be the whole sequence (positions 0..T-1), which is
        the uncached reference path: every step recomputes K/V for all previous tokens."""
        B, T = input_ids.shape
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + T, device=input_ids.device)
        cos, sin = self.rope(positions)
        attend = ATTENTION[self.attn_name]
        rep = self.n_heads // self.n_kv

        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin       # RoPE on queries and keys only, never on values
            k = k * cos + _rotate_half(k) * sin
            if cache is not None:
                k, v = cache.write(i, k, v, start)    # k/v now cover positions 0 .. start+T-1
            # grouped-query attention: each K/V head serves `rep` query heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            o = attend(q, k, v, causal=True, q_offset=start)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if cache is not None:
            cache.length = start + T
        x = self.final_norm(x if all_logits else x[:, -1:])
        return self.lm_head(x).float()                # [B, T or 1, vocab]
