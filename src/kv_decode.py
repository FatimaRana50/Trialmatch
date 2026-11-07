"""Greedy decoding loops on top of ManualQwen2, with timing and memory measurement.

greedy_nocache : reference path. Every step runs the model over the whole sequence so far
                 (O(T^2) work over a generation); no K/V is reused.
greedy_cached  : prefill writes K/V for the prompt once; each later step feeds one token
                 and appends its K/V to the cache.
greedy_hf      : Hugging Face model.generate (greedy), an external reference for token equality.
"""
import time

import torch

from manual_decoder import ManualQwen2


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mem_reset(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)
    return 0


def mem_peak(device, base):
    return (torch.cuda.max_memory_allocated(device) - base) / 2**20 if device.type == "cuda" else None


def _top2_gap(logits):
    top = torch.topk(logits, 2, dim=-1).values
    return float(top[0, 0] - top[0, 1])


def _result(tokens, gaps, t0, t1, t2, peak, extra=None):
    n = len(tokens)
    r = {"tokens": tokens, "top2_gaps": gaps, "n_new": n,
         "prefill_s": t1 - t0, "decode_s": t2 - t1, "total_s": t2 - t0,
         # decode throughput excludes the first token, which comes out of prefill
         "decode_tok_s": (n - 1) / (t2 - t1) if n > 1 and t2 > t1 else None,
         "e2e_tok_s": n / (t2 - t0), "peak_mem_mb": peak}
    r.update(extra or {})
    return r


@torch.no_grad()
def greedy_cached(mm: ManualQwen2, input_ids, max_new, eos_ids=(), stop_at_eos=True):
    dev = input_ids.device
    base = mem_reset(dev)
    sync(dev); t0 = time.perf_counter()
    cache = mm.new_cache(input_ids.shape[1] + max_new)
    logits = mm.forward(input_ids, cache)[:, -1]          # prefill: whole prompt in one pass
    nxt = logits.argmax(-1)
    tokens, gaps = [int(nxt)], [_top2_gap(logits)]
    sync(dev); t1 = time.perf_counter()
    while len(tokens) < max_new and not (stop_at_eos and tokens[-1] in eos_ids):
        logits = mm.forward(nxt.view(1, 1), cache)[:, -1]  # one new token, attends to the cache
        nxt = logits.argmax(-1)
        tokens.append(int(nxt)); gaps.append(_top2_gap(logits))
    sync(dev); t2 = time.perf_counter()
    return _result(tokens, gaps, t0, t1, t2, mem_peak(dev, base),
                   {"cache_mb": cache.nbytes() / 2**20, "cache_len": cache.length})


@torch.no_grad()
def greedy_nocache(mm: ManualQwen2, input_ids, max_new, eos_ids=(), stop_at_eos=True):
    dev = input_ids.device
    base = mem_reset(dev)
    sync(dev); t0 = time.perf_counter()
    seq = input_ids
    logits = mm.forward(seq)[:, -1]                       # same work as a prefill
    nxt = logits.argmax(-1)
    tokens, gaps = [int(nxt)], [_top2_gap(logits)]
    sync(dev); t1 = time.perf_counter()
    while len(tokens) < max_new and not (stop_at_eos and tokens[-1] in eos_ids):
        seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
        logits = mm.forward(seq)[:, -1]                   # recompute everything from position 0
        nxt = logits.argmax(-1)
        tokens.append(int(nxt)); gaps.append(_top2_gap(logits))
    sync(dev); t2 = time.perf_counter()
    return _result(tokens, gaps, t0, t1, t2, mem_peak(dev, base))


@torch.no_grad()
def greedy_hf(model, input_ids, max_new, eos_ids=(), stop_at_eos=True, pad_id=None):
    from transformers import GenerationConfig
    dev = input_ids.device
    cfg = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                           max_new_tokens=max_new, min_new_tokens=1 if stop_at_eos else max_new,
                           eos_token_id=list(eos_ids) or None, pad_token_id=pad_id)
    base = mem_reset(dev)
    sync(dev); t0 = time.perf_counter()
    out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=cfg)
    sync(dev); t2 = time.perf_counter()
    tokens = out[0, input_ids.shape[1]:].tolist()
    r = _result(tokens, [], t0, t0, t2, mem_peak(dev, base))
    r["prefill_s"] = r["decode_s"] = r["decode_tok_s"] = None   # generate() does not expose the split
    return r


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
