"""Generate the LaTeX tables in report/tables/ directly from results/ and logs/, so every number
in the paper comes from a result file.   python src/make_report_tables.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "tables")
SHORT = {"included": "incl.", "not included": "not incl.", "excluded": "excl.", "not excluded": "not excl.",
         "not enough information": "NEI", "not applicable": "N/A"}


def load(path):
    return json.load(open(os.path.join(ROOT, path)))


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(text)
    print("wrote", name)


def f3(x):
    return f"{x:.3f}"


def main_results():
    rows = []
    for split in ("val", "test"):
        for tag, name in (("base", "Zero-shot"), ("lora", "LoRA")):
            m = load(f"results/{tag}_{split}_metrics.json")
            o, i, e = m["overall"], m["inclusion"], m["exclusion"]
            rows.append(f"{split.title()} & {name} & {f3(o['classification']['accuracy'])} & "
                        f"{f3(o['classification']['macro_f1'])} & {f3(i['classification']['accuracy'])} & "
                        f"{f3(e['classification']['accuracy'])} & {f3(o['evidence']['micro_f1'])} & "
                        f"{f3(o['evidence']['macro_f1'])} \\\\")
        if split == "val":
            rows.append("\\midrule")
    write("main_results.tex", "\n".join(rows) + "\n")


def per_class():
    m = load("results/lora_test_metrics.json")
    b = load("results/base_test_metrics.json")
    rows = []
    for lab, c in m["overall"]["classification"]["per_class"].items():
        cb = b["overall"]["classification"]["per_class"][lab]
        rows.append(f"{lab} & {c['support']} & {cb['f1']:.2f} & {c['precision']:.2f} & {c['recall']:.2f} & {c['f1']:.2f} \\\\")
    write("per_class.tex", "\n".join(rows) + "\n")


def confusion(part):
    c = load("results/lora_test_metrics.json")[part]["classification"]["confusion"]
    cols = c["cols_pred"]
    head = "Gold $\\downarrow$ / Pred $\\rightarrow$ & " + " & ".join(SHORT.get(x, x) for x in cols) + " \\\\"
    body = [f"{SHORT.get(g, g)} & " + " & ".join(str(v) for v in row) + " \\\\" for g, row in zip(c["rows_gold"], c["matrix"])]
    spec = "l" + "r" * len(cols)
    write(f"cm_{part}.tex", f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n" + "\n".join(body) +
          "\n\\bottomrule\n\\end{tabular}\n")


def evidence():
    rows = []
    for tag, name in (("base", "Zero-shot"), ("lora", "LoRA")):
        m = load(f"results/{tag}_test_metrics.json")
        for part in ("overall", "inclusion", "exclusion"):
            e = m[part]["evidence"]
            rows.append(f"{name} & {part} & {e['micro_precision']:.2f} & {e['micro_recall']:.2f} & {e['micro_f1']:.2f} & "
                        f"{e['macro_precision']:.2f} & {e['macro_recall']:.2f} & {e['macro_f1']:.2f} & "
                        f"{e['empty_gold_correctly_empty']:.2f} \\\\")
        if tag == "base":
            rows.append("\\midrule")
    write("evidence.tex", "\n".join(rows) + "\n")


def design():
    rows = []
    for tag, name, cfg in (("lora", "LoRA (selected)", "logs/train_config.json"),
                           ("lora_os", "LoRA + oversampling", "logs/train_config_os.json")):
        m = load(f"results/{tag}_val_metrics.json")
        c = load(cfg)
        pc = m["overall"]["classification"]["per_class"]
        rare = "/".join(f"{pc[l]['f1']:.2f}" for l in ("excluded", "not included", "not applicable"))
        rows.append(f"{name} & {c['train_examples']} & {c['best_val_loss']:.3f} & {f3(m['overall']['classification']['accuracy'])} & "
                    f"{f3(m['overall']['classification']['macro_f1'])} & {f3(m['overall']['evidence']['micro_f1'])} & {rare} \\\\")
    write("design.tex", "\n".join(rows) + "\n")


def kv():
    rows = []
    data = load("results/kv_bench.json")["rows"]
    by = {(r["input_len"], r["output_len"], r["mode"]): r for r in data}
    for L in sorted({r["input_len"] for r in data}):
        for N in sorted({r["output_len"] for r in data}):
            nc, kc, hf = by[(L, N, "no_cache")], by[(L, N, "kv_cache")], by[(L, N, "hf_generate")]
            rows.append(f"{L} & {N} & {nc['prefill_s']*1000:.0f} / {kc['prefill_s']*1000:.0f} & "
                        f"{nc['decode_s']:.2f} / {kc['decode_s']:.2f} & {nc['e2e_tok_s']:.1f} / {kc['e2e_tok_s']:.1f} / {hf['e2e_tok_s']:.1f} & "
                        f"{nc['total_s']/kc['total_s']:.1f}$\\times$ & {nc['peak_mem_mb']:.0f} / {kc['peak_mem_mb']:.0f} \\\\")
    write("kv_bench.tex", "\n".join(rows) + "\n")
    c = load("results/kv_correctness.json")
    fixed = [x for x in c["checks"] if isinstance(x["output_len"], int)]
    real = [x for x in c["checks"] if isinstance(x["output_len"], str) and x["pair"].startswith("cached")]
    attn = [x for x in c["checks"] if x["pair"].startswith("attn_")]
    summary = {
        "fixed_cached_vs_nocache": f"{sum(x['equal'] for x in fixed if x['pair']=='cached_vs_nocache')}/{sum(1 for x in fixed if x['pair']=='cached_vs_nocache')}",
        "fixed_cached_vs_hf": f"{sum(x['equal'] for x in fixed if x['pair']=='cached_vs_hf_generate')}/{sum(1 for x in fixed if x['pair']=='cached_vs_hf_generate')}",
        "real_cached_vs_nocache": f"{sum(x['equal'] for x in real if x['pair']=='cached_vs_nocache')}/{sum(1 for x in real if x['pair']=='cached_vs_nocache')}",
        "real_cached_vs_hf": f"{sum(x['equal'] for x in real if x['pair']=='cached_vs_hf_generate')}/{sum(1 for x in real if x['pair']=='cached_vs_hf_generate')}",
        "attn_variants": f"{sum(x['equal'] for x in attn)}/{len(attn)}",
        "total": f"{c['passed']}/{c['total']}",
    }
    print("kv correctness", summary)


def attention():
    data = load("results/attention_bench.json")["rows"]
    lens = sorted({r["seq_len"] for r in data})
    impls = list(dict.fromkeys(r["impl"] for r in data))
    by = {(r["seq_len"], r["impl"]): r for r in data}
    rows = []
    for im in impls:
        cells = []
        for L in lens:
            r = by[(L, im)]
            cells.append(f"{r['ms']:.2f} / {r['peak_mem_mb']:.0f}" if "ms" in r else "n/a")
        rows.append(im.replace("sdpa: ", "SDPA ").replace(" (PyTorch, full matrix)", " (full matrix)")
                    .replace(" (PyTorch)", "") + " & " + " & ".join(cells) + " \\\\")
    write("attn_bench.tex", "\n".join(rows) + "\n")
    c = load("results/attention_correctness.json")
    for dt in ("float32", "float16"):
        cs = [x for x in c["cases"] if x["dtype"] == dt]
        print(dt, "cases", len(cs), "pass", sum(x["pass"] for x in cs),
              "tiled max", max(x["tiled_max_err"] for x in cs), "tiled mean", max(x["tiled_mean_err"] for x in cs))


def spec():
    data = load("results/spec_bench.json")
    rows = []
    for r in data["rows"]:
        if r["method"] != "speculative":
            continue
        rows.append(f"{r['setting'][0]} & {r['draft'].replace('0.5B ', '')} & {r['k']} & {r['acceptance_rate']:.2f} & "
                    f"{r['mean_accepted_per_round']:.2f} & {r['target_forwards']}/{r['greedy_target_forwards']} & "
                    f"{r['tok_s']:.1f} & {r['speedup']:.2f}$\\times$ & {r['equal_outputs']}/{r['n_prompts']} \\\\")
    base = {r["setting"][0]: r for r in data["rows"] if r["method"] != "speculative"}
    write("spec_bench.tex", "\n".join(rows) + "\n")
    print("greedy baselines", {k: (round(v["tok_s"], 1), round(v["total_s"], 1), v["tokens"]) for k, v in base.items()},
          "equal", data["equal"], "/", data["total"])


if __name__ == "__main__":
    main_results(); per_class(); confusion("inclusion"); confusion("exclusion"); evidence(); design(); kv(); attention(); spec()
