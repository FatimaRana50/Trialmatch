"""Task 5: speculative decoding vs ordinary greedy decoding of the target.

Setting A (screening): target = fine-tuned Qwen2.5-1.5B (LoRA merged), real val screening prompts,
                       stop at <|im_end|>, max 64 new tokens. Drafts: base Qwen2.5-0.5B and,
                       if --draft-adapter is given, a 0.5B fine-tuned with the same recipe.
Setting B (free text): target = base Qwen2.5-1.5B, draft = base Qwen2.5-0.5B,
                       "summarise this note" prompts, max 128 new tokens (longer outputs).

  python src/bench_spec.py --target-adapter /content/drive/MyDrive/trialmatch/lora_r16/final_adapter \
                           [--draft-adapter /content/drive/MyDrive/trialmatch/draft_0.5b/final_adapter]

Writes results/spec_bench.json and results/spec_bench.png.
"""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import data
import infer
from kv_decode import first_mismatch, greedy_cached
from manual_decoder import ManualQwen2
from speculative import speculative_greedy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def check_compatible(tok_t, tok_d, m_t, m_d):
    """Speculative decoding compares token IDs directly, so both models must share the vocabulary."""
    same_vocab = tok_t.get_vocab() == tok_d.get_vocab()
    same_rows = m_t.lm_head.out_features == m_d.lm_head.out_features
    info = {"same_tokenizer_vocab": same_vocab, "vocab_size": len(tok_t),
            "lm_head_rows_target": m_t.lm_head.out_features, "lm_head_rows_draft": m_d.lm_head.out_features,
            "target_params_M": round(sum(p.numel() for p in m_t.parameters()) / 1e6),
            "draft_params_M": round(sum(p.numel() for p in m_d.parameters()) / 1e6)}
    assert same_vocab and same_rows, f"draft/target incompatible: {info}"
    return info


def encode(tok, messages, dev):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)


def run_setting(name, target, drafts, prompts, max_new, ks, eos, tok):
    rows, checks = [], []
    # warm-up both paths once
    greedy_cached(target, prompts[0], 8, eos)
    for d in drafts.values():
        speculative_greedy(target, d, prompts[0], 8, 2, eos)
    base = [greedy_cached(target, p, max_new, eos) for p in prompts]
    tot = lambda rs, key: sum(r[key] for r in rs)
    g_time, g_tok = tot(base, "total_s"), tot(base, "n_new")
    rows.append({"setting": name, "method": "greedy (target only)", "draft": None, "k": None,
                 "total_s": g_time, "tokens": g_tok, "tok_s": g_tok / g_time, "speedup": 1.0,
                 "peak_mem_mb": max(r["peak_mem_mb"] or 0 for r in base)})
    print(rows[-1])
    for dname, d in drafts.items():
        for k in ks:
            spec = [speculative_greedy(target, d, p, max_new, k, eos) for p in prompts]
            eq = [first_mismatch(b["tokens"], s["tokens"]) for b, s in zip(base, spec)]
            for i, (b, s, m) in enumerate(zip(base, spec, eq)):
                c = {"setting": name, "draft": dname, "k": k, "prompt": i, "equal": m is None, "first_mismatch": m}
                if m is not None and m < len(b["top2_gaps"]):
                    c["greedy_top2_gap_at_mismatch"] = b["top2_gaps"][m]
                    c["greedy_text"] = tok.decode(b["tokens"]); c["spec_text"] = tok.decode(s["tokens"])
                checks.append(c)
            s_time, s_tok = tot(spec, "total_s"), tot(spec, "n_new")
            row = {"setting": name, "method": "speculative", "draft": dname, "k": k,
                   "total_s": s_time, "tokens": s_tok, "tok_s": s_tok / s_time, "speedup": g_time / s_time,
                   "acceptance_rate": tot(spec, "accepted") / max(1, tot(spec, "drafted")),
                   "mean_accepted_per_round": tot(spec, "accepted") / max(1, tot(spec, "rounds")),
                   "draft_s": tot(spec, "draft_s"), "verify_s": tot(spec, "verify_s"),
                   "target_forwards": tot(spec, "target_forwards"), "greedy_target_forwards": g_tok,
                   "equal_outputs": sum(m is None for m in eq), "n_prompts": len(prompts),
                   "peak_mem_mb": max(r["peak_mem_mb"] or 0 for r in spec)}
            rows.append(row)
            print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()})
    return rows, checks


def hf_assisted_reference(model_t, model_d, prompts, max_new, eos, pad_id):
    """Library reference only: HF assisted generation should also equal plain greedy."""
    from transformers import GenerationConfig
    cfg = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new, eos_token_id=eos,
                           pad_token_id=pad_id, repetition_penalty=1.0)
    res = []
    for p in prompts:
        try:
            a = model_t.generate(input_ids=p, attention_mask=torch.ones_like(p), assistant_model=model_d,
                                 generation_config=cfg)[0, p.shape[1]:].tolist()
            g = model_t.generate(input_ids=p, attention_mask=torch.ones_like(p),
                                 generation_config=cfg)[0, p.shape[1]:].tolist()
            res.append(a == g)
        except Exception as e:                       # API differences between versions
            return {"error": str(e)[:200]}
    return {"hf_assisted_equals_hf_greedy": sum(res), "n": len(res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=infer.BASE_MODEL)
    ap.add_argument("--target-adapter", required=True)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--draft-adapter", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--n-prompts", type=int, default=20)
    args = ap.parse_args()

    rows_all = data.load_rows()
    manifest = json.load(open(os.path.join(ROOT, "data", "split_manifest.json")))
    val = data.rows_for(rows_all, manifest, "val")[: args.n_prompts]

    # ---------------- setting A: screening prompts, fine-tuned target
    tok, m_t = infer.load_model(args.target, args.target_adapter)
    dev = m_t.device
    tok_d, m_d = infer.load_model(args.draft)
    compat = check_compatible(tok, tok_d, m_t, m_d)
    print(compat)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    target = ManualQwen2(m_t)
    drafts = {"0.5B base": ManualQwen2(m_d)}
    if args.draft_adapter:
        _, m_d_ft = infer.load_model(args.draft, args.draft_adapter)
        drafts["0.5B fine-tuned"] = ManualQwen2(m_d_ft)
    prompts_a = [encode(tok, data.row_messages(r), dev) for r in val]
    rows, checks = run_setting("A: screening JSON (fine-tuned 1.5B target)", target, drafts,
                               prompts_a, 64, args.ks, eos, tok)
    hf_ref = hf_assisted_reference(m_t, m_d, prompts_a[:5], 64, eos, tok.pad_token_id)
    print("HF assisted reference:", hf_ref)
    del target, drafts, m_t
    gc.collect(); torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---------------- setting B: longer free-text generation, base target
    _, m_tb = infer.load_model(args.target)
    target = ManualQwen2(m_tb)
    prompts_b = [encode(tok, [{"role": "user", "content":
                               "Summarise this patient note in three sentences.\n\n" + r["note"]}], dev)
                 for r in val]
    rows_b, checks_b = run_setting("B: free-text summary (base 1.5B target)", target,
                                   {"0.5B base": ManualQwen2(m_d)}, prompts_b, 128, args.ks, eos, tok)
    rows += rows_b; checks += checks_b

    n_eq = sum(c["equal"] for c in checks)
    print(f"\nspeculative == greedy on {n_eq}/{len(checks)} prompt runs")
    env = {"target": args.target, "target_adapter": args.target_adapter, "draft": args.draft,
           "draft_adapter": args.draft_adapter, "dtype": "float16" if dev.type == "cuda" else "float32",
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "attention": "sdpa (explicit offset causal mask)", "batch": 1, "compatibility": compat,
           "hf_assisted_reference": hf_ref}
    json.dump({"env": env, "rows": rows, "equal": n_eq, "total": len(checks), "checks": checks},
              open(os.path.join(ROOT, "results", "spec_bench.json"), "w"), indent=2)
    plot(rows, os.path.join(ROOT, "results", "spec_bench.png"))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (setting, draft) in dict.fromkeys((r["setting"], r["draft"]) for r in rows if r["method"] == "speculative"):
        pts = sorted((r["k"], r["speedup"], r["acceptance_rate"]) for r in rows
                     if r["setting"] == setting and r["draft"] == draft)
        lab = f"{setting.split(':')[0]} / {draft}"
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=lab)
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("draft length k"); axes[0].set_ylabel("speedup vs greedy (x)")
    axes[1].set_xlabel("draft length k"); axes[1].set_ylabel("draft acceptance rate")
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
