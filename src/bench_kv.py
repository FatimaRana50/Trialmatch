"""Task 3: correctness tests and benchmark for KV-cached vs uncached greedy decoding.

  python src/bench_kv.py                       # full run (T4: ~10 min)
  python src/bench_kv.py --quick               # small smoke test

Writes results/kv_correctness.json, results/kv_bench.json, results/kv_bench.png.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import data
import infer
from kv_decode import first_mismatch, greedy_cached, greedy_hf, greedy_nocache
from manual_decoder import ManualQwen2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def text_prompt(tok, rows, length, device):
    """Exactly `length` tokens of real patient-note text (val split), for controlled lengths."""
    text = "\n\n".join(r["note"] for r in rows)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < length:
        ids = ids + ids
    return torch.tensor([ids[:length]], device=device)


def chat_prompt(tok, row, device):
    """The real screening prompt used by the application."""
    text = tok.apply_chat_template(data.row_messages(row), tokenize=False, add_generation_prompt=True)
    return torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=device)


def compare(name, ref, other):
    i = first_mismatch(ref["tokens"], other["tokens"])
    d = {"pair": name, "equal": i is None, "n_ref": len(ref["tokens"]), "n_other": len(other["tokens"]),
         "first_mismatch": i}
    if i is not None and i < len(ref["top2_gaps"]):
        # a small top-1/top-2 logit gap at the divergence step means a near tie flipped by rounding
        d["ref_top2_gap_at_mismatch"] = ref["top2_gaps"][i]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=infer.BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--in-lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--out-lens", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "reference", "tiled"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.in_lens, args.out_lens, args.repeats = [64, 256], [16, 48], 1

    tok, model = infer.load_model(args.base, args.adapter)
    dev = model.device
    mm = ManualQwen2(model, attn=args.attn)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    rows = data.load_rows()
    manifest = json.load(open(os.path.join(ROOT, "data", "split_manifest.json")))
    val = data.rows_for(rows, manifest, "val")
    env = {"model": args.base, "adapter": args.adapter, "dtype": str(mm.dtype), "attention": args.attn,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch": torch.__version__, "decoding": "greedy (argmax), batch 1",
           "kv_bytes_per_token": 2 * len(mm.layers) * mm.n_kv * mm.head_dim * torch.finfo(mm.dtype).bits // 8}
    print(env)

    # ---------------- correctness: fixed lengths (EOS ignored) + real screening prompts
    checks = []
    for L in args.in_lens:
        for N in args.out_lens:
            ids = text_prompt(tok, val, L, dev)
            nc = greedy_nocache(mm, ids, N, stop_at_eos=False)
            kc = greedy_cached(mm, ids, N, stop_at_eos=False)
            hf = greedy_hf(model, ids, N, stop_at_eos=False, pad_id=tok.pad_token_id)
            for c in (compare("cached_vs_nocache", nc, kc), compare("cached_vs_hf_generate", hf, kc)):
                checks.append({"input_len": L, "output_len": N, **c})
                print(checks[-1])
    for r in val[:5]:
        ids = chat_prompt(tok, r, dev)
        nc = greedy_nocache(mm, ids, 64, eos)
        kc = greedy_cached(mm, ids, 64, eos)
        hf = greedy_hf(model, ids, 64, eos, pad_id=tok.pad_token_id)
        for c in (compare("cached_vs_nocache", nc, kc), compare("cached_vs_hf_generate", hf, kc)):
            checks.append({"input_len": ids.shape[1], "output_len": "eos<=64",
                           "annotation_id": r["annotation_id"], **c})
            print(checks[-1])
        print("   output:", tok.decode(kc["tokens"], skip_special_tokens=True))
    # ---------------- attention variants inside the real model (cache on, one prompt)
    ids = chat_prompt(tok, val[0], dev)
    by_attn = {}
    for name in ("sdpa", "reference", "tiled"):
        by_attn[name] = greedy_cached(ManualQwen2(model, attn=name), ids, 48, eos)
    for name in ("reference", "tiled"):
        checks.append({"input_len": ids.shape[1], "output_len": "eos<=48",
                       **compare(f"attn_{name}_vs_sdpa", by_attn["sdpa"], by_attn[name])})
        print(checks[-1])

    n_eq = sum(c["equal"] for c in checks)
    print(f"\ntoken-equality checks passed: {n_eq}/{len(checks)}")
    json.dump({"env": env, "passed": n_eq, "total": len(checks), "checks": checks},
              open(os.path.join(ROOT, "results", "kv_correctness.json"), "w"), indent=2)

    # ---------------- benchmark: warm-up + repeated synchronized timing, medians reported
    bench = []
    for L in args.in_lens:
        ids = text_prompt(tok, val, L, dev)
        for N in args.out_lens:
            for mode, fn in (("no_cache", lambda: greedy_nocache(mm, ids, N, stop_at_eos=False)),
                             ("kv_cache", lambda: greedy_cached(mm, ids, N, stop_at_eos=False)),
                             ("hf_generate", lambda: greedy_hf(model, ids, N, stop_at_eos=False,
                                                               pad_id=tok.pad_token_id))):
                fn()                                      # warm-up (kernel selection, allocator)
                runs = [fn() for _ in range(args.repeats)]
                med = lambda k: statistics.median(r[k] for r in runs) if runs[0][k] is not None else None
                row = {"input_len": L, "output_len": N, "mode": mode, "repeats": args.repeats,
                       "prefill_s": med("prefill_s"), "decode_s": med("decode_s"), "total_s": med("total_s"),
                       "decode_tok_s": med("decode_tok_s"), "e2e_tok_s": med("e2e_tok_s"),
                       "peak_mem_mb": max(r["peak_mem_mb"] or 0 for r in runs),
                       "cache_mb": runs[0].get("cache_mb")}
                bench.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    json.dump({"env": env, "rows": bench}, open(os.path.join(ROOT, "results", "kv_bench.json"), "w"), indent=2)
    plot(bench, os.path.join(ROOT, "results", "kv_bench.png"))


def plot(bench, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, style in (("no_cache", "--"), ("kv_cache", "-"), ("hf_generate", ":")):
        for L in sorted({b["input_len"] for b in bench}):
            pts = sorted((b["output_len"], b["e2e_tok_s"], b["peak_mem_mb"]) for b in bench
                         if b["mode"] == mode and b["input_len"] == L)
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", label=f"{mode} in={L}")
            axes[1].plot([p[0] for p in pts], [p[2] for p in pts], style, marker="o", label=f"{mode} in={L}")
    axes[0].set_xlabel("output tokens"); axes[0].set_ylabel("end-to-end tokens/s")
    axes[1].set_xlabel("output tokens"); axes[1].set_ylabel("peak extra memory (MiB)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3); axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
