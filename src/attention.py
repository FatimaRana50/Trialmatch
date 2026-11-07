"""Scaled dot-product attention: a reference implementation and a tiled
online-softmax (FlashAttention-style) implementation, written in plain PyTorch.

Shapes: q [B, H, Tq, D], k/v [B, H, Tk, D].
Positions: query i is at absolute position q_offset + i, key j at position j.
With a KV cache, Tk = q_offset + Tq, so the causal rule "key position <= query position"
is a mask offset by q_offset. (torch's is_causal=True aligns the mask top-left and is
wrong for a chunk of queries that starts after position 0, so masks here are explicit.)
"""
import math

import torch
import torch.nn.functional as F


def causal_mask(tq, tk, q_offset, device):
    """True where attention is NOT allowed (key in the future of the query)."""
    q_pos = torch.arange(q_offset, q_offset + tq, device=device)
    k_pos = torch.arange(tk, device=device)
    return k_pos[None, :] > q_pos[:, None]


def reference_attention(q, k, v, causal=True, q_offset=0, scale=None):
    """softmax(QK^T * scale + mask) V, materialising the full Tq x Tk score matrix.
    Computed in fp32 and cast back to the input dtype."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)          # stable softmax: largest exponent is exp(0)=1
    p = s.exp()
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal=True, q_offset=0, scale=None, block_q=64, block_k=64):
    """Same result as reference_attention, but keys/values are processed in blocks of
    block_k with a running max m, running normaliser l and running output acc per query,
    so no Tq x Tk matrix is ever stored (only block_q x block_k at a time).

    For each new key block with scores s:
        m_new = max(m, rowmax(s))
        alpha = exp(m - m_new)                # rescales everything accumulated so far
        p     = exp(s - m_new)
        l     = alpha * l + rowsum(p)
        acc   = alpha * acc + p @ V_block
    and at the end out = acc / l.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs), float("-inf"), device=q.device)
        l = torch.zeros(B, H, qe - qs, device=q.device)
        acc = torch.zeros(B, H, qe - qs, D, device=q.device)
        # with a causal mask, key blocks entirely after the last query in this tile are skipped
        k_end = min(Tk, q_offset + qe) if causal else Tk
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = torch.matmul(qb, k[:, :, ks:ke].float().transpose(-1, -2))
            if causal:
                q_pos = torch.arange(q_offset + qs, q_offset + qe, device=q.device)
                k_pos = torch.arange(ks, ke, device=q.device)
                s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            # a row whose keys so far are all masked has m_new = -inf; use 0 so exp() gives 0, not NaN
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)
            p = torch.exp(s - m_safe[..., None])
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + torch.matmul(p, v[:, :, ks:ke].float())
            m = m_new
        out[:, :, qs:qe] = acc / l[..., None]
    return out.to(q.dtype)


def sdpa_attention(q, k, v, causal=True, q_offset=0, scale=None):
    """PyTorch's fused scaled_dot_product_attention, with an explicit offset-aware mask."""
    mask = None
    if causal:
        mask = ~causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device)  # SDPA bool mask: True = keep
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


ATTENTION = {"reference": reference_attention, "tiled": tiled_attention, "sdpa": sdpa_attention}
