"""TrialMatch browser application (Gradio).

  ./run_app.sh                 (or: python app/app.py)  -> http://127.0.0.1:7860
Environment variables (optional): TRIALMATCH_ADAPTER, TRIALMATCH_DRAFT_ADAPTER (local path or HF Hub id),
PORT, HOST, TRIALMATCH_SHARE=1 (public Gradio link), TRIALMATCH_INBROWSER=1 (open the browser).
"""
import html
import inspect
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
EX_BY_NAME = {f"{e['patient_id']} · {e['trial_id']} · {e['trial_title'][:60]}": e for e in EXAMPLES}
REPORT_DIR = os.path.join(tempfile.gettempdir(), "trialmatch_reports")

LABEL_STYLE = {  # text colour, background
    "included": ("#116329", "#dafbe1"), "not excluded": ("#116329", "#dafbe1"),
    "not included": ("#a40e26", "#ffebe9"), "excluded": ("#a40e26", "#ffebe9"),
    "not enough information": ("#7d4e00", "#fff8c5"), "not applicable": ("#57606a", "#eaeef2"),
    "unparseable output": ("#6639ba", "#fbefff"),
}
VERDICT_STYLE = {"Likely": ("#a40e26", "#ffebe9"), "Undetermined": ("#7d4e00", "#fff8c5"),
                 "Potentially": ("#116329", "#dafbe1")}

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important}
#hero {padding: 18px 22px; border-radius: 14px; background: linear-gradient(135deg,#eef2ff,#f8fafc);
       border: 1px solid #e0e7ff; margin-bottom: 6px}
#hero h1 {margin: 0; font-size: 26px; color: #1e1b4b}
#hero p {margin: 4px 0 0; color: #475569}
.notice {background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:10px 14px; color:#78350f; font-size:14px}
#status {font-size: 14px; padding: 8px 12px; border-radius: 10px; background: #f1f5f9; border: 1px solid #e2e8f0}
.step-title {font-weight: 700; color: #312e81; font-size: 15px; margin-bottom: 4px}
.placeholder {color:#64748b; text-align:center; padding:36px 10px; border:2px dashed #e2e8f0; border-radius:12px}
.tm-card {background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; margin:10px 0;
          box-shadow: 0 1px 2px rgba(15,23,42,.05)}
.tm-head {display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap}
.tm-type {font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:#475569}
.tm-crit {margin:8px 0; color:#0f172a; font-size:15px}
.tm-badge {display:inline-block; border-radius:999px; padding:2px 12px; font-weight:600; font-size:13px; white-space:nowrap}
.tm-ev {background:#fefce8; border-left:4px solid #facc15; padding:6px 10px; margin:6px 0; border-radius:4px; font-size:14px; color:#1f2937}
.tm-expl {color:#334155; font-size:14px; margin-top:8px}
.tm-meta {color:#64748b; font-size:12px; margin-top:8px}
.tm-verdict {border-radius:12px; padding:16px 18px; margin:6px 0 4px}
.tm-verdict .v {font-size:20px; font-weight:800}
.tm-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:10px; margin-top:10px}
.tm-box {background:#ffffffaa; border:1px solid #e2e8f0; border-radius:10px; padding:10px 12px; font-size:14px}
.tm-box b {color:#0f172a}
.tm-box ul {margin:6px 0 0 18px; padding:0}
.tm-note div {padding:2px 0; font-size:14px; color:#1f2937}
"""

# always open in light mode, whatever the operating-system theme is
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", radius_size="md")


def _badge(label):
    fg, bg = LABEL_STYLE.get(label, ("#334155", "#f1f5f9"))
    return f"<span class='tm-badge' style='color:{fg};background:{bg}'>{html.escape(label)}</span>"


def render_results(res, example):
    e = html.escape
    expert = {c["text"].strip(): c for c in example["criteria"]} if example else {}
    o = res["overall"]
    fg, bg = next((v for k, v in VERDICT_STYLE.items() if o["verdict"].startswith(k)), ("#334155", "#f1f5f9"))
    lst = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs) or "<li>none</li>"
    counts = " · ".join(f"{e(k)}: {v}" for k, v in o["counts"].items())
    top = (f"<div class='tm-verdict' style='background:{bg};border:1px solid {fg}33'>"
           f"<div class='v' style='color:{fg}'>{e(o['verdict'])}</div>"
           f"<div style='color:#334155;margin-top:4px'>{e(o['reason'])}</div>"
           f"<div class='tm-meta'>Rule-based summary of the criterion-level predictions ({counts}); "
           f"not a ground-truth label. Rules: {e('; '.join(o['rules']))}.</div>"
           f"<div class='tm-grid'>"
           f"<div class='tm-box'><b>Indicating ineligibility ({len(o['failing'])})</b><ul>{lst(o['failing'])}</ul></div>"
           f"<div class='tm-box'><b>Missing information ({len(o['missing_information'])})</b><ul>{lst(o['missing_information'])}</ul></div>"
           f"<div class='tm-box'><b>Decisions without cited evidence ({len(o['unsupported_decisions'])})</b>"
           f"<ul>{lst(o['unsupported_decisions'])}</ul></div></div></div>")
    cards, agree, n_exp = [], 0, 0
    for n, r in enumerate(res["results"]):
        ev = "".join(f"<div class='tm-ev'><b>[{i}]</b> {e(t)}</div>" for i, t in zip(r["evidence_ids"], r["evidence_text"]))
        ex = expert.get(r["criterion"].strip())
        exp_html = ""
        if ex:
            n_exp += 1
            ok = ex["expert_label"] == r["label"]
            agree += ok
            exp_html = (f"<div class='tm-meta'>Dataset expert label: {_badge(ex['expert_label'])} "
                        f"&nbsp;expert evidence IDs {ex['expert_evidence'] or '[]'} &nbsp;"
                        f"<b style='color:{'#116329' if ok else '#a40e26'}'>{'✔ agrees' if ok else '✘ differs'}</b></div>")
        meta = f"{r['prompt_tokens']} prompt tokens"
        if r.get("prefill_tokens") is not None and r["prefill_tokens"] != r["prompt_tokens"]:
            meta += f" ({r['prompt_tokens'] - r['prefill_tokens']} reused from the shared note cache)"
        meta += f" · {r['new_tokens']} generated · {r['seconds']} s"
        if r.get("acceptance_rate") is not None:
            meta += f" · draft acceptance {r['acceptance_rate']:.0%}"
        cards.append(
            f"<div class='tm-card'><div class='tm-head'><span class='tm-type'>{n + 1}. {e(r['criterion_type'])} criterion</span>"
            f"{_badge(r['label'])}</div><div class='tm-crit'>{e(r['criterion'])}</div>"
            f"{ev or '<div class=tm-meta>No supporting sentence cited.</div>'}"
            f"<div class='tm-expl'>{e(r['explanation'])}</div>{exp_html}<div class='tm-meta'>{meta}</div></div>")
    agree_html = (f"<div class='tm-meta' style='font-size:13px'>Agreement with the dataset's expert labels on this "
                  f"example: <b>{agree}/{n_exp}</b></div>" if n_exp else "")
    note = "".join(f"<div><b>[{i}]</b> {e(s)}</div>" for i, s in res["sentences"])
    return (top + agree_html + "".join(cards) +
            f"<details class='tm-card'><summary><b>Patient note as numbered sentences</b></summary>"
            f"<div class='tm-note' style='margin-top:8px'>{note}</div></details>")


def results_table(res):
    return [[n + 1, r["criterion_type"], r["criterion"], r["label"], ", ".join(map(str, r["evidence_ids"])),
             " | ".join(r["evidence_text"]), r["seconds"]] for n, r in enumerate(res["results"])]


PLACEHOLDER = "<div class='placeholder'>Results will appear here after you run the screening.</div>"


# ------------------------------------------------------------------ callbacks

def model_status():
    icon = "✅" if S.MODELS.ready else ("⚠️" if S.MODELS.status.startswith("Model failed") else "⏳")
    return f"{icon} **Model status:** {S.MODELS.status}", gr.Timer(active=not S.MODELS.ready
                                                                     and not S.MODELS.status.startswith("Model failed"))


def load_example(name):
    ex = EX_BY_NAME.get(name)
    if not ex:
        raise gr.Error("Choose an example from the list first.")
    return ex["note"], S.criteria_to_text([{"type": c["type"], "text": c["text"]} for c in ex["criteria"]]), ex


def fetch_trial(nct):
    try:
        title, text = S.fetch_ctgov(nct)
    except S.InputError as e:
        raise gr.Error(str(e))
    gr.Info(f"Loaded criteria for {nct.strip().upper()}: {title[:80]}")
    return text, None


def run_screening(note, criteria, mode, k, reuse, example, progress=gr.Progress()):
    try:
        res = S.screen(note, criteria, mode, int(k), bool(reuse), progress=progress)
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


def clear_all():
    return "", "", None, PLACEHOLDER, [], None, None


def clear_example_if_edited(note, example):
    return example if example and note == example["note"] else None


def mode_changed(mode):
    spec, kv = S.MODES[mode] == "speculative", S.MODES[mode] == "kv_cache"
    return gr.update(visible=spec), gr.update(visible=kv)


def run_compare(note, criterion, ctype, modes, k):
    if not modes:
        raise gr.Error("Select at least one decoding mechanism.")
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
        raise gr.Error("There are no criteria in the Screening tab yet.")
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
- KV-cache mode reuses the patient-note prefix across criteria (prefilled once per screening).

### Held-out test results (8 unseen patients, 206 criteria)
{table}

### Known limitations
- Never predicts the rare labels *not applicable* / *not included* / *excluded* on the test split
  (8–28 training examples each); leans to *not excluded* for exclusion criteria.
- Trained on 37 patients from TREC-style case reports; real clinical notes are longer and messier.
- Absence of a statement is never treated as evidence; such criteria are reported as missing information.

**{S.DISCLAIMER}**
"""


def _progress_kw(event, target):
    """Show the progress indicator on one component only (Gradio 5.x+), so spinners do not stack."""
    return {"show_progress_on": [target]} if "show_progress_on" in inspect.signature(event).parameters else {}


# ------------------------------------------------------------------ layout

# Gradio 6 moved css/theme/js from Blocks() to launch(); support both 5.x and 6.x
_STYLE = {"css": CSS, "theme": THEME, "js": FORCE_LIGHT_JS}
_GR6 = int(gr.__version__.split(".")[0]) >= 6

with gr.Blocks(title="TrialMatch screening prototype", **({} if _GR6 else _STYLE)) as demo:
    example_state = gr.State(None)
    gr.HTML("<div id='hero'><h1>🩺 TrialMatch</h1><p>Criterion-level clinical-trial eligibility screening with a "
            "fine-tuned Qwen2.5-1.5B model. Every decision cites the patient-note sentences it relies on.</p></div>")
    gr.HTML(f"<div class='notice'><b>⚠ Educational prototype.</b> {html.escape(S.DISCLAIMER)}</div>")
    status = gr.Markdown(elem_id="status")
    timer = gr.Timer(2.0)

    with gr.Tab("Screen a patient"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<div class='step-title' style='padding:10px 12px 0'>1 · Patient note</div>")
                    with gr.Row():
                        ex_dd = gr.Dropdown(list(EX_BY_NAME), label="Load an example (validation split, has expert labels)",
                                            scale=4)
                        ex_btn = gr.Button("Load", scale=1, min_width=80)
                    note = gr.Textbox(label="Patient note", lines=13, show_label=False,
                                      placeholder="Paste a clinical note. Sentences are numbered automatically "
                                                  "(or keep your own '0. ...' numbering, one sentence per line).")
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<div class='step-title' style='padding:10px 12px 0'>2 · Trial eligibility criteria</div>")
                    with gr.Row():
                        nct = gr.Textbox(label="ClinicalTrials.gov ID (optional)", placeholder="e.g. NCT01397994", scale=4)
                        nct_btn = gr.Button("Fetch", scale=1, min_width=80)
                    criteria = gr.Textbox(label="Criteria", lines=13, show_label=False,
                                          placeholder="Inclusion Criteria:\n- Age 18 years or older\n- ...\n\n"
                                                      "Exclusion Criteria:\n- Pregnancy\n- ...")
        with gr.Group():
            gr.HTML("<div class='step-title' style='padding:10px 12px 0'>3 · Inference settings</div>")
            with gr.Row():
                mode = gr.Radio(list(S.MODES), value=list(S.MODES)[0], label="Decoding mechanism", scale=3)
                with gr.Column(scale=1, min_width=220):
                    k = gr.Slider(1, 6, value=4, step=1, label="Draft length k", visible=False)
                    reuse = gr.Checkbox(value=True, label="Reuse the patient-note KV cache across criteria (faster)")
        with gr.Row():
            run_btn = gr.Button("▶ Run screening", variant="primary", size="lg", scale=4)
            clear_btn = gr.Button("Clear", size="lg", scale=1)

        gr.HTML("<div class='step-title' style='margin-top:14px'>Results</div>")
        out_html = gr.HTML(PLACEHOLDER)
        out_files = gr.File(label="Download the screening report (HTML + JSON, with provenance)", file_count="multiple")
        with gr.Accordion("Table view", open=False):
            out_table = gr.Dataframe(headers=["#", "type", "criterion", "label", "sentence IDs", "evidence", "seconds"],
                                     wrap=True, interactive=False)
        with gr.Accordion("Raw JSON result", open=False):
            out_json = gr.JSON()

    with gr.Tab("Inference comparison"):
        gr.Markdown("Runs **one** criterion through the selected decoding mechanisms live on this machine and checks "
                    "that each one produces exactly the same tokens as KV-cache greedy decoding.")
        with gr.Row():
            cmp_crit = gr.Textbox(label="Criterion", lines=2, scale=3)
            cmp_type = gr.Radio(["inclusion", "exclusion"], value="inclusion", label="Type", scale=1)
        cmp_fill = gr.Button("Use the note and the first criterion from the Screening tab")
        cmp_modes = gr.CheckboxGroup(list(S.MODES), value=list(S.MODES)[:3], label="Mechanisms to compare")
        cmp_k = gr.Slider(1, 6, value=4, step=1, label="Draft length k (speculative)")
        cmp_btn = gr.Button("▶ Run comparison", variant="primary")
        cmp_out = gr.Dataframe(headers=["mode", "prompt tokens", "new tokens", "prefill s", "decode s", "total s",
                                        "tokens/s", "peak extra MiB", "draft acceptance", "same tokens as KV greedy",
                                        "output"], wrap=True, interactive=False)
        gr.Markdown("### Attention check (Task 4), live\nRandom Q/K/V with 12 heads × 128 dims; error against a float64 reference.")
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

    timer.tick(model_status, None, [status, timer], queue=False, show_progress="hidden")
    demo.load(model_status, None, [status, timer], queue=False, show_progress="hidden")
    ex_btn.click(load_example, ex_dd, [note, criteria, example_state])
    nct_btn.click(fetch_trial, nct, [criteria, example_state])
    note.input(clear_example_if_edited, [note, example_state], example_state)
    mode.change(mode_changed, mode, [k, reuse])
    run_btn.click(run_screening, [note, criteria, mode, k, reuse, example_state],
                  [out_html, out_table, out_files, out_json], **_progress_kw(run_btn.click, out_html))
    clear_btn.click(clear_all, None, [note, criteria, example_state, out_html, out_table, out_files, out_json])
    cmp_fill.click(first_criterion, criteria, [cmp_crit, cmp_type])
    cmp_btn.click(run_compare, [note, cmp_crit, cmp_type, cmp_modes, cmp_k], cmp_out,
                  **_progress_kw(cmp_btn.click, cmp_out))
    att_btn.click(run_attention_check, [att_len, att_causal], att_out)


if __name__ == "__main__":
    S.MODELS.preload()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("TRIALMATCH_SHARE") == "1", allowed_paths=[REPORT_DIR],
        inbrowser=os.environ.get("TRIALMATCH_INBROWSER") == "1", **(_STYLE if _GR6 else {}))
