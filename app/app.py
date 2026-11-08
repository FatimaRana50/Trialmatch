"""TrialMatch browser application (Gradio).

  python app/app.py            -> open http://127.0.0.1:7860
Environment variables (optional): TRIALMATCH_ADAPTER, TRIALMATCH_DRAFT_ADAPTER (local path or HF Hub id),
PORT, TRIALMATCH_SHARE=1 (public Gradio link).
"""
import html
import json
import math
import os
import tempfile
import time
import traceback

import gradio as gr
import torch

import screening as S

ROOT = S.ROOT
EXAMPLES = json.load(open(os.path.join(os.path.dirname(__file__), "examples.json")))["examples"]
EX_BY_NAME = {f"{e['name']}: {e['trial_title'][:70]}": e for e in EXAMPLES}
REPORT_DIR = os.path.join(tempfile.gettempdir(), "trialmatch_reports")

LABEL_COLOR = {"included": "#1a7f37", "not excluded": "#1a7f37", "not included": "#cf222e", "excluded": "#cf222e",
               "not enough information": "#bf8700", "not applicable": "#6e7781", "unparseable output": "#8250df"}

CSS = """
.notice {background:#fff4e5;border:1px solid #f0a020;border-radius:8px;padding:10px 14px;color:#5a3b00}
.card {border:1px solid var(--border-color-primary);border-radius:8px;padding:10px 14px;margin:8px 0}
.lab {display:inline-block;color:#fff;border-radius:12px;padding:1px 10px;font-weight:600;font-size:13px}
.ev {background:rgba(255,212,0,.25);border-left:3px solid #e3b300;padding:2px 8px;margin:4px 0;font-size:14px}
.muted {opacity:.75;font-size:13px}
.verdict {font-size:18px;font-weight:700}
"""


def _badge(label):
    return f"<span class='lab' style='background:{LABEL_COLOR.get(label, '#555')}'>{html.escape(label)}</span>"


def render_results(res, example):
    e = html.escape
    expert = {}
    if example:
        expert = {c["text"].strip(): c for c in example["criteria"]}
    o = res["overall"]
    lst = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs) or "<li>none</li>"
    top = (f"<div class='card'><div class='verdict'>{e(o['verdict'])}</div><div>{e(o['reason'])}</div>"
           f"<div class='muted'>Rule-based aggregation of criterion-level predictions, not a ground-truth label. "
           f"Rules: {e('; '.join(o['rules']))}.</div>"
           f"<details><summary>Criteria indicating ineligibility ({len(o['failing'])})</summary><ul>{lst(o['failing'])}</ul></details>"
           f"<details open><summary>Missing information ({len(o['missing_information'])})</summary><ul>{lst(o['missing_information'])}</ul></details>"
           f"<details><summary>Decisions without cited evidence ({len(o['unsupported_decisions'])})</summary>"
           f"<ul>{lst(o['unsupported_decisions'])}</ul></details></div>")
    cards = []
    agree = n_exp = 0
    for n, r in enumerate(res["results"]):
        ev = "".join(f"<div class='ev'><b>[{i}]</b> {e(t)}</div>" for i, t in zip(r["evidence_ids"], r["evidence_text"]))
        ex = expert.get(r["criterion"].strip())
        exp_html = ""
        if ex:
            n_exp += 1
            agree += ex["expert_label"] == r["label"]
            exp_html = (f"<div class='muted'>Dataset expert label: {_badge(ex['expert_label'])} "
                        f"evidence IDs {ex['expert_evidence'] or '[]'} "
                        f"{'✔ agrees' if ex['expert_label'] == r['label'] else '✘ differs'}</div>")
        extra = f" · draft acceptance {r['acceptance_rate']:.0%}" if r.get("acceptance_rate") is not None else ""
        cards.append(
            f"<div class='card'><div><b>{n + 1}. {e(r['criterion_type'].title())}</b> {_badge(r['label'])}</div>"
            f"<div style='margin:6px 0'>{e(r['criterion'])}</div>"
            f"{ev or '<div class=muted>No supporting sentence cited.</div>'}"
            f"<div style='margin-top:6px'><i>{e(r['explanation'])}</i></div>{exp_html}"
            f"<div class='muted'>{r['prompt_tokens']} prompt tokens · {r['new_tokens']} generated · {r['seconds']} s{extra}</div></div>")
    agree_html = (f"<div class='muted'>Agreement with dataset expert labels on this example: {agree}/{n_exp}</div>"
                  if n_exp else "")
    note = "".join(f"<div><b>[{i}]</b> {e(s)}</div>" for i, s in res["sentences"])
    return (top + agree_html + "".join(cards) +
            f"<details><summary>Patient note as numbered sentences</summary><div class='card'>{note}</div></details>")


def results_table(res):
    return [[n + 1, r["criterion_type"], r["criterion"], r["label"], ", ".join(map(str, r["evidence_ids"])),
             " | ".join(r["evidence_text"]), r["seconds"]] for n, r in enumerate(res["results"])]


# ------------------------------------------------------------------ callbacks

def load_example(name):
    ex = EX_BY_NAME.get(name)
    if not ex:
        raise gr.Error("Choose an example first.")
    return ex["note"], S.criteria_to_text([{"type": c["type"], "text": c["text"]} for c in ex["criteria"]]), ex


def fetch_trial(nct):
    try:
        title, text = S.fetch_ctgov(nct)
    except S.InputError as e:
        raise gr.Error(str(e))
    gr.Info(f"Loaded criteria for {nct.strip().upper()}: {title[:80]}")
    return text, None


def run_screening(note, criteria, mode, k, example, progress=gr.Progress()):
    try:
        res = S.screen(note, criteria, mode, int(k), progress=progress)
    except S.InputError as e:
        raise gr.Error(str(e))
    except torch.OutOfMemoryError:
        raise gr.Error("Out of memory. Try fewer criteria or a shorter note.")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Screening failed: {type(e).__name__}: {e}")
    # an example's expert labels only apply while its exact criteria are being screened
    if example and S.criteria_to_text([{"type": c["type"], "text": c["text"]} for c in example["criteria"]]) != criteria:
        example = None
    h, j = S.save_reports(res, REPORT_DIR)
    return render_results(res, example), results_table(res), [h, j], res


def clear_example_if_edited(note, example):
    return example if example and note == example["note"] else None


def run_compare(note, criterion, ctype, modes, k):
    if not modes:
        raise gr.Error("Select at least one decoding mode.")
    try:
        rows = S.compare_modes(note, criterion, ctype, modes, int(k))
    except S.InputError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Comparison failed: {type(e).__name__}: {e}")
    fmt = lambda v: round(v, 4) if isinstance(v, float) else ("—" if v is None else v)
    cols = list(rows[0].keys())
    return [[fmt(r[c]) for c in cols] for r in rows]


def first_criterion(criteria):
    crits = S.parse_criteria(criteria)
    if not crits:
        raise gr.Error("No criteria in the Screening tab yet.")
    return crits[0]["text"], crits[0]["type"]


def run_attention_check(seq_len, causal):
    from attention import reference_attention, sdpa_attention, tiled_attention
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, H, D = int(seq_len), 12, 128
    g = torch.Generator().manual_seed(int(time.time()) % 10000)
    q, k, v = (torch.randn(1, H, T, D, generator=g).to(dev) for _ in range(3))
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(D)
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=dev).triu(1), float("-inf"))
    gt = torch.softmax(s, -1) @ v.double()
    rows = []
    for name, fn in (("reference (full matrix)", lambda: reference_attention(q, k, v, causal)),
                     ("tiled online-softmax (64x64 blocks)", lambda: tiled_attention(q, k, v, causal)),
                     ("torch SDPA (fused)", lambda: sdpa_attention(q, k, v, causal))):
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter(); out = fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        err = (out.double() - gt).abs()
        rows.append([name, f"{err.max().item():.2e}", f"{err.mean().item():.2e}", round((time.perf_counter() - t) * 1000, 2)])
    return rows


def stored_tables():
    def load(name):
        p = os.path.join(ROOT, "results", name)
        return json.load(open(p)) if os.path.exists(p) else None
    out = []
    kv = load("kv_bench.json")
    out.append([[r["input_len"], r["output_len"], r["mode"], r["prefill_s"] and round(r["prefill_s"], 3),
                 round(r["total_s"], 2), round(r["e2e_tok_s"], 1), round(r["peak_mem_mb"])] for r in kv["rows"]] if kv else [])
    sp = load("spec_bench.json")
    out.append([[r["setting"][:2], r["draft"] or "—", r["k"] or "—", round(r["tok_s"], 1), round(r["speedup"], 2),
                 round(r["acceptance_rate"], 2) if r.get("acceptance_rate") is not None else "—"]
                for r in sp["rows"]] if sp else [])
    at = load("attention_bench.json")
    out.append([[r["seq_len"], r["impl"], round(r["ms"], 2) if "ms" in r else "n/a",
                 round(r["peak_mem_mb"], 1) if r.get("peak_mem_mb") is not None else "n/a"] for r in at["rows"]] if at else [])
    return out


def model_card():
    def m(name):
        p = os.path.join(ROOT, "results", name)
        return json.load(open(p)) if os.path.exists(p) else None
    rows = []
    for tag, nm in (("base", "Qwen2.5-1.5B-Instruct zero-shot"), ("lora", "+ LoRA fine-tuned (this app)")):
        r = m(f"{tag}_test_metrics.json")
        if r:
            o = r["overall"]
            rows.append(f"| {nm} | {o['classification']['accuracy']:.3f} | {o['classification']['macro_f1']:.3f} | "
                        f"{o['evidence']['micro_f1']:.3f} |")
    table = ("| Model | Accuracy | Macro-F1 | Evidence micro-F1 |\n|---|---|---|---|\n" + "\n".join(rows)) if rows else ""
    return f"""
### Model
- Target: **{S.TARGET_BASE}** with LoRA adapter `{S.TARGET_ADAPTER}` (r=16, all attention + MLP projections).
- Draft for speculative decoding: **{S.DRAFT_BASE}** (same tokenizer), LoRA adapter `{S.DRAFT_ADAPTER}`.
- Output: `{{"evidence": [sentence IDs], "label": ...}}`, greedy decoding, max {S.MAX_NEW_TOKENS} new tokens.
- Explanations are templates filled from the predicted label and the cited sentences only.

### Held-out test results (8 unseen patients, 206 criteria)
{table}

### Known limitations
- Never predicts the rare labels *not applicable* / *not included* / *excluded* on the test split
  (8–28 training examples each); leans to *not excluded* for exclusion criteria.
- Trained on 37 patients from TREC-style case reports; real clinical notes are longer and messier.
- Absence of a statement is never treated as evidence; such criteria are reported as missing information.

**{S.DISCLAIMER}**
"""


# ------------------------------------------------------------------ layout

# Gradio 6 moved css/theme from Blocks() to launch(); support both 5.x and 6.x
_STYLE = {"css": CSS, "theme": gr.themes.Soft()}
_GR6 = int(gr.__version__.split(".")[0]) >= 6

with gr.Blocks(title="TrialMatch screening prototype", **({} if _GR6 else _STYLE)) as demo:
    example_state = gr.State(None)
    gr.Markdown("# TrialMatch: criterion-level clinical trial screening")
    gr.HTML(f"<div class='notice'><b>⚠ Educational prototype.</b> {html.escape(S.DISCLAIMER)}</div>")

    with gr.Tab("Screen a patient"):
        with gr.Row():
            with gr.Column(scale=1):
                ex_dd = gr.Dropdown(list(EX_BY_NAME), label="Example (validation split, with expert labels)")
                ex_btn = gr.Button("Load example")
                note = gr.Textbox(label="Patient note", lines=12,
                                  placeholder="Paste a clinical note. Sentences are numbered automatically "
                                              "(or keep your own '0. ...' numbering, one sentence per line).")
            with gr.Column(scale=1):
                with gr.Row():
                    nct = gr.Textbox(label="ClinicalTrials.gov ID (optional)", placeholder="NCT01397994", scale=3)
                    nct_btn = gr.Button("Fetch criteria", scale=1)
                criteria = gr.Textbox(label="Eligibility criteria (one per line)", lines=12,
                                      placeholder="Inclusion Criteria:\n- Age 18 years or older\n- ...\n\n"
                                                  "Exclusion Criteria:\n- Pregnancy\n- ...")
                mode = gr.Radio(list(S.MODES), value=list(S.MODES)[0], label="Decoding mechanism")
                k = gr.Slider(1, 6, value=4, step=1, label="Draft length k (speculative mode only)")
        run_btn = gr.Button("Run screening", variant="primary")
        out_html = gr.HTML()
        out_table = gr.Dataframe(headers=["#", "type", "criterion", "label", "sentence IDs", "evidence", "seconds"],
                                 label="Criterion-level results", wrap=True, interactive=False)
        out_files = gr.File(label="Download screening report (HTML and JSON, with provenance)", file_count="multiple")
        with gr.Accordion("Raw JSON result", open=False):
            out_json = gr.JSON()

    with gr.Tab("Inference comparison"):
        gr.Markdown("Runs **one** criterion through the selected decoding mechanisms live on this machine "
                    "and checks that each produces the same tokens as KV-cache greedy decoding.")
        with gr.Row():
            cmp_crit = gr.Textbox(label="Criterion", lines=2, scale=3)
            cmp_type = gr.Radio(["inclusion", "exclusion"], value="inclusion", label="Type", scale=1)
        cmp_fill = gr.Button("Use the first criterion and the note from the Screening tab")
        cmp_modes = gr.CheckboxGroup(list(S.MODES), value=list(S.MODES)[:3], label="Mechanisms to compare")
        cmp_k = gr.Slider(1, 6, value=4, step=1, label="Draft length k")
        cmp_btn = gr.Button("Run comparison", variant="primary")
        cmp_out = gr.Dataframe(headers=["mode", "prompt tokens", "new tokens", "prefill s", "decode s", "total s",
                                        "tokens/s", "peak extra MiB", "draft acceptance", "same tokens as KV greedy",
                                        "output"], wrap=True, interactive=False)
        gr.Markdown("### Attention check (Task 4), live\nRandom Q/K/V with 12 heads × 128 dims; error vs a float64 reference.")
        with gr.Row():
            att_len = gr.Slider(16, 2048, value=256, step=16, label="Sequence length")
            att_causal = gr.Checkbox(value=True, label="Causal mask")
            att_btn = gr.Button("Run attention check")
        att_out = gr.Dataframe(headers=["implementation", "max abs error", "mean abs error", "ms"], interactive=False)
        gr.Markdown("### Recorded benchmarks (Tesla T4, fp16; from `results/`)")
        t_kv, t_sp, t_at = stored_tables()
        gr.Dataframe(t_kv, headers=["input", "output", "mode", "prefill s", "total s", "tokens/s", "peak MiB"],
                     label="KV cache vs no cache vs HF generate", interactive=False)
        gr.Dataframe(t_sp, headers=["setting", "draft", "k", "tokens/s", "speedup", "acceptance"],
                     label="Speculative decoding (A: screening JSON, B: free-text summary)", interactive=False)
        gr.Dataframe(t_at, headers=["seq len", "implementation", "ms", "peak MiB"],
                     label="Attention implementations", interactive=False)

    with gr.Tab("About"):
        gr.Markdown(model_card())

    ex_btn.click(load_example, ex_dd, [note, criteria, example_state])
    nct_btn.click(fetch_trial, nct, [criteria, example_state])
    note.input(clear_example_if_edited, [note, example_state], example_state)
    run_btn.click(run_screening, [note, criteria, mode, k, example_state], [out_html, out_table, out_files, out_json])
    cmp_fill.click(first_criterion, criteria, [cmp_crit, cmp_type])
    cmp_btn.click(run_compare, [note, cmp_crit, cmp_type, cmp_modes, cmp_k], cmp_out)
    att_btn.click(run_attention_check, [att_len, att_causal], att_out)
    demo.load(lambda: S.MODELS.load_target() and None)        # load the model when the page first opens


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("TRIALMATCH_SHARE") == "1", allowed_paths=[REPORT_DIR], **(_STYLE if _GR6 else {}))
