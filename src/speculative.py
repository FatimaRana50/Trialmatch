"""Greedy speculative decoding (draft model proposes, target model verifies).

Both models run through ManualQwen2 with their own KVCache. Invariant between rounds:
`seq` holds prompt + all committed tokens; the last committed token x has been produced by
the target but is not yet in either cache ("pending"). The draft cache may lag one more
token behind (after a fully accepted round), so the draft always feeds seq[dc.length:].

One round with k drafted tokens:
  1. draft:  feed pending tokens, then greedily propose d1..dk (k small forwards)
  2. verify: target runs ONE forward over [x, d1..dk] -> greedy predictions p0..pk,
             where p_i is the target's next token after the i-th input
  3. accept: n = length of the prefix with d_{i+1} == p_i
  4. commit: d1..dn and then p_n (the target's correction if n < k, a free bonus token if n == k)
  5. caches: target keeps x, d1..dn (crop away rejected d's); draft is cropped likewise
Every committed token is a target greedy prediction given the committed prefix, so the
output equals ordinary greedy decoding of the target (up to fp rounding, see bench_spec.py).
"""
import time

import torch

from kv_decode import mem_peak, mem_reset, sync
from manual_decoder import ManualQwen2


@torch.no_grad()
def speculative_greedy(target: ManualQwen2, draft: ManualQwen2, input_ids, max_new, k=4,
                       eos_ids=(), stop_at_eos=True):
    dev = input_ids.device
    base = mem_reset(dev)
    sync(dev); t0 = time.perf_counter()
    max_len = input_ids.shape[1] + max_new + k + 2
    tc, dc = target.new_cache(max_len), draft.new_cache(max_len)
    first = int(target.forward(input_ids, tc)[0, -1].argmax())   # target prefill -> first token
    draft.forward(input_ids, dc)                                    # draft prefill (logits unused)
    seq = input_ids[0].tolist() + [first]
    out = [first]
    sync(dev); t1 = time.perf_counter()

    rounds = drafted = accepted = 0
    draft_s = verify_s = 0.0
    accepted_per_round = []
    while len(out) < max_new and not (stop_at_eos and out[-1] in eos_ids):
        kk = min(k, max_new - len(out))
        # 1. draft
        ta = time.perf_counter()
        props, inp = [], seq[dc.length:]
        for _ in range(kk):
            d = int(draft.forward(torch.tensor([inp], device=dev), dc)[0, -1].argmax())
            props.append(d)
            inp = [d]
        sync(dev); tb = time.perf_counter()
        # 2. verify all proposals in one target forward
        n_before = len(seq)                                   # includes pending x
        ver_in = seq[tc.length:] + props                      # [x, d1..dk]
        preds = target.forward(torch.tensor([ver_in], device=dev), tc, all_logits=True)[0].argmax(-1).tolist()
        sync(dev); tc_ = time.perf_counter()
        # 3./4. accept matching prefix, then the target's own token
        n = 0
        while n < kk and props[n] == preds[n]:
            n += 1
        new = props[:n] + [preds[n]]
        # 5. roll caches back to the committed prefix (x, d1..dn); the new pending token is preds[n]
        tc.crop(n_before + n)
        dc.crop(min(dc.length, n_before + n))
        seq += new
        out += new
        rounds += 1; drafted += kk; accepted += n; accepted_per_round.append(n)
        draft_s += tb - ta; verify_s += tc_ - tb
        if stop_at_eos:
            for i, t in enumerate(new):
                if t in eos_ids:
                    out = out[:len(out) - len(new) + i + 1]
                    break
    out = out[:max_new]
    sync(dev); t2 = time.perf_counter()
    return {"tokens": out, "n_new": len(out), "k": k,
            "prefill_s": t1 - t0, "decode_s": t2 - t1, "total_s": t2 - t0,
            "draft_s": draft_s, "verify_s": verify_s,
            "e2e_tok_s": len(out) / (t2 - t0),
            "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "mean_accepted_per_round": accepted / rounds if rounds else None,
            "accepted_per_round": accepted_per_round,
            "target_forwards": rounds + 1, "peak_mem_mb": mem_peak(dev, base)}
