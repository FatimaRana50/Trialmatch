"""Task 4: numerical validation and benchmark of reference vs tiled online-softmax attention.

  python src/bench_attention.py            # correctness + benchmark
  python src/bench_attention.py --quick    # small run (also works on CPU)

Ground truth is a float64 softmax(QK^T/sqrt(d) + mask)V. Writes results/attention_correctness.json,
results/attention_bench.json and results/attention_bench.png.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

from attention import causal_mask, reference_attention, sdpa_attention, tiled_attention

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ground_truth(q, k, v, causal, q_offset):
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    return torch.softmax(s, dim=-1) @ v.double()


def correctness(device):
    # (B, H, Tq, Tk, D, causal, q_offset, block): includes non-divisible lengths/blocks,
    # cache-style queries (Tq < Tk, q_offset = Tk - Tq) and single-token decode (Tq = 1)
    cases = [
        (1, 1, 8, 8, 16, True, 0, 4),
        (1, 2, 17, 17, 32, True, 0, 5),
        (2, 4, 64, 64, 64, True, 0, 16),
        (2, 4, 64, 64, 64, False, 0, 16),
        (1, 12, 100, 100, 128, True, 0, 32),
        (1, 12, 128, 128, 128, True, 0, 64),
        (1, 8, 33, 97, 64, True, 64, 16),       # chunk of 33 queries continuing after 64 cached keys
        (1, 12, 1, 300, 128, True, 299, 64),    # one decode step over a 300-token cache
        (1, 2, 50, 70, 32, False, 0, 7),        # cross-attention style, Tq != Tk, no mask
    ]
    if device.type == "cuda":
        cases.append((1, 12, 1024, 1024, 128, True, 0, 128))
    dtypes = [torch.float32] + ([torch.float16] if device.type == "cuda" else [])
    out = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for (B, H, Tq, Tk, D, causal, off, blk) in cases:
        q = torch.randn(B, H, Tq, D, generator=g)
        k = torch.randn(B, H, Tk, D, generator=g)
        v = torch.randn(B, H, Tk, D, generator=g)
        # one case with large logits stresses the running-max rescaling
        if (Tq, Tk) == (17, 17):
            q = q * 30
        for dt in dtypes:
            qd, kd, vd = (t.to(device, dt) for t in (q, k, v))
            gt = ground_truth(qd, kd, vd, causal, off)
            res = {"shape": [B, H, Tq, Tk, D], "causal": causal, "q_offset": off, "block": blk,
                   "dtype": str(dt).replace("torch.", "")}
            for name, fn in (("reference", lambda: reference_attention(qd, kd, vd, causal, off)),
                             ("tiled", lambda: tiled_attention(qd, kd, vd, causal, off, block_q=blk, block_k=blk)),
                             ("sdpa", lambda: sdpa_attention(qd, kd, vd, causal, off))):
                err = (fn().double() - gt).abs()
                res[f"{name}_max_err"] = float(err.max())
                res[f"{name}_mean_err"] = float(err.mean())
            tol = 1e-4 if dt == torch.float32 else 2e-2
            res["tolerance"] = tol
            res["pass"] = all(res[f"{n}_max_err"] < tol for n in ("reference", "tiled", "sdpa"))
            out.append(res)
            print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in res.items()})
    return out


def timed(fn, device, repeats):
    fn()                                                  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else None
    return statistics.median(times) * 1000, peak


def benchmark(device, lens, repeats, dtype):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    H, D = 12, 128                                        # Qwen2.5-1.5B attention heads / head dim
    rows = []
    for T in lens:
        q, k, v = (torch.randn(1, H, T, D, device=device, dtype=dtype) for _ in range(3))
        # queries start at position 0 here, so is_causal=True is the correct mask and lets the
        # fused backends run (they do not accept an arbitrary attn_mask)
        fused = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        variants = {
            "reference (PyTorch, full matrix)": lambda: reference_attention(q, k, v, True),
            "tiled online-softmax (PyTorch)": lambda: tiled_attention(q, k, v, True, block_q=128, block_k=128),
            "sdpa: math backend": ("math", fused),
            "sdpa: memory-efficient backend": ("eff", fused),
            "sdpa: flash backend": ("flash", fused),
        }
        backend = {"math": SDPBackend.MATH, "eff": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}
        for name, fn in variants.items():
            row = {"seq_len": T, "heads": H, "head_dim": D, "dtype": str(dtype).replace("torch.", ""), "impl": name}
            try:
                if isinstance(fn, tuple):
                    b, f = fn
                    with sdpa_kernel(backend[b]):
                        row["ms"], row["peak_mem_mb"] = timed(f, device, repeats)
                else:
                    row["ms"], row["peak_mem_mb"] = timed(fn, device, repeats)
            except (RuntimeError, torch.OutOfMemoryError) as e:   # e.g. flash needs sm80+, T4 is sm75
                row["error"] = str(e).split("\n")[0][:160]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            print(row)
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for impl in dict.fromkeys(r["impl"] for r in rows):
        pts = [(r["seq_len"], r["ms"], r.get("peak_mem_mb")) for r in rows if r["impl"] == impl and "ms" in r]
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
        if pts[0][2] is not None:
            axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=impl)
    for ax, yl in zip(axes, ("time per call (ms)", "peak extra memory (MiB)")):
        ax.set_xlabel("sequence length"); ax.set_ylabel(yl); ax.set_xscale("log", base=2)
        ax.set_yscale("log"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quick:
        args.lens, args.repeats = [128, 256], 3

    checks = correctness(device)
    n = sum(c["pass"] for c in checks)
    print(f"\nattention correctness: {n}/{len(checks)} passed")
    env = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"env": env, "passed": n, "total": len(checks), "cases": checks},
              open(os.path.join(ROOT, "results", "attention_correctness.json"), "w"), indent=2)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    rows = benchmark(device, args.lens, args.repeats, dtype)
    json.dump({"env": env, "rows": rows}, open(os.path.join(ROOT, "results", "attention_bench.json"), "w"), indent=2)
    plot(rows, os.path.join(ROOT, "results", "attention_bench.png"))


if __name__ == "__main__":
    main()
