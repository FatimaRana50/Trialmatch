"""Screening engine behind the browser app: input handling, model inference with the
selectable decoding mechanisms, rule-based aggregation, and report export."""
import datetime
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

import data
import infer
from kv_decode import first_mismatch, greedy_cached, greedy_hf, greedy_nocache
from manual_decoder import ManualQwen2
from speculative import speculative_greedy

TARGET_BASE = os.environ.get("TRIALMATCH_BASE", infer.BASE_MODEL)
TARGET_ADAPTER = os.environ.get("TRIALMATCH_ADAPTER", "f777777777/trialmatch-qwen2.5-1.5b-lora")
DRAFT_BASE = os.environ.get("TRIALMATCH_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
DRAFT_ADAPTER = os.environ.get("TRIALMATCH_DRAFT_ADAPTER", "f777777777/trialmatch-qwen2.5-0.5b-draft-lora")

MODES = {
    "KV cache (manual decoder)": "kv_cache",
    "Speculative (0.5B draft + 1.5B target)": "speculative",
    "Hugging Face generate, batched (library reference)": "hf_generate",
    "No cache (slow reference)": "no_cache",
}
MAX_NOTE_TOKENS = 2000
MAX_CRITERIA = 40
MAX_NEW_TOKENS = 64
HF_BATCH = 8
DISCLAIMER = ("Educational screening prototype only. It is NOT a clinical enrollment decision and must not be "
              "used for patient care. All outputs must be reviewed by qualified clinical research staff.")


class InputError(ValueError):
    """Problems with the user's input, shown to the user as a readable message."""


# ------------------------------------------------------------------ models (lazy, shared)

def cpu_dtype(n_params_b, extra_gb=1.5):
    """fp32 on CPU when there is room (bf16 matmuls are slow without AVX512-BF16), else bf16 at half
    the memory. TRIALMATCH_DTYPE=float32|bfloat16 overrides."""
    forced = os.environ.get("TRIALMATCH_DTYPE")
    if forced:
        return getattr(torch, forced)
    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    return torch.float32 if avail_gb > n_params_b * 4 + extra_gb else torch.bfloat16


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


class Models:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = self.target_hf = self.target = None
        self.draft_hf = self.draft = None
        self.draft_name = None
        self.status = "Model not loaded yet."
        self.ready = False

    def load_target(self):
        with self.lock:
            if self.target is None:
                self.status = (f"Loading {TARGET_BASE} + fine-tuned adapter... "
                               "(the first start downloads about 3 GB; this can take several minutes)")
                dtype = torch.float16 if torch.cuda.is_available() else cpu_dtype(1.55)
                if dtype == torch.bfloat16:
                    self.status += f" Low free memory ({free_gb():.1f} GB): using bfloat16 (slower on CPU)."
                try:
                    self.tok, self.target_hf = infer.load_model(TARGET_BASE, TARGET_ADAPTER or None, dtype=dtype)
                except Exception as e:
                    self.status = f"Model failed to load: {type(e).__name__}: {e}"
                    raise
                self.target = ManualQwen2(self.target_hf)
                dev = self.target_hf.device
                where = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
                self.status = (f"Ready: {TARGET_BASE.split('/')[-1]} + LoRA on {where} "
                               f"({str(self.target.dtype).replace('torch.', '')})")
                if self.target.dtype == torch.bfloat16 and dev.type == "cpu":
                    self.status += (" · low-memory mode: close other programs and restart the app for "
                                    "faster float32 inference")
                self.ready = True
        return self.tok, self.target_hf, self.target

    def load_draft(self):
        self.load_target()
        with self.lock:
            if self.draft is None:
                dtype = self.target.dtype
                if self.target_hf.device.type == "cpu":
                    # the draft only proposes tokens and the target verifies every one, so a bf16 draft
                    # (~1 GB instead of ~2 GB) keeps the output identical; it can only change acceptance
                    if dtype == torch.float32 and free_gb() < 0.5 * 4 + 0.8:
                        dtype = torch.bfloat16
                    if free_gb() < 0.5 * 2 + 0.5:
                        raise InputError(f"Not enough free memory ({free_gb():.1f} GB) to also load the 0.5B draft "
                                         "model for speculative decoding. Close other programs (e.g. Chrome) or "
                                         "use another mode.")
                try:
                    tok_d, self.draft_hf = infer.load_model(DRAFT_BASE, DRAFT_ADAPTER or None, dtype=dtype)
                    self.draft_name = f"{DRAFT_BASE} + {DRAFT_ADAPTER}" if DRAFT_ADAPTER else DRAFT_BASE
                except Exception:                     # fine-tuned draft unavailable -> base draft
                    tok_d, self.draft_hf = infer.load_model(DRAFT_BASE, dtype=dtype)
                    self.draft_name = DRAFT_BASE
                assert tok_d.get_vocab() == self.tok.get_vocab(), "draft tokenizer differs from target"
                self.draft = ManualQwen2(self.draft_hf)
        return self.draft

    def preload(self):
        """Start loading the target in the background so the page opens immediately."""
        def run():
            try:
                self.load_target()
            except Exception:
                pass                                  # status already holds the error
        threading.Thread(target=run, daemon=True).start()

    def device(self):
        return self.target_hf.device if self.target_hf is not None else None


MODELS = Models()


# ------------------------------------------------------------------ input handling

_NUMBERED = re.compile(r"^\s*\d+\.\s")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(note):
    """Already-numbered notes ('0. ...' per line, like the dataset) keep their IDs.
    Otherwise split on line breaks and sentence ends and number from 0."""
    note = (note or "").strip()
    lines = [l for l in note.splitlines() if l.strip()]
    if lines and all(_NUMBERED.match(l) for l in lines):
        sents = data.split_note(note)
        if [i for i, _ in sents] == list(range(len(sents))):
            return sents
    out = []
    for line in lines:
        out += [s.strip() for s in _SENT_SPLIT.split(line.strip()) if s.strip()]
    return list(enumerate(out))


_HEADER = re.compile(r"^\s*(inclusion|exclusion)\s*(criteria)?\s*:?\s*$", re.I)
_PREFIX = re.compile(r"^\s*(I|E|inclusion|exclusion)\s*[:|]\s*", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·]+|\d+[.)]|[a-z][.)])\s+")


def parse_criteria(text, default_type="inclusion"):
    """One criterion per line. Lines 'Inclusion Criteria:' / 'Exclusion Criteria:' switch the type
    for the lines below (the ClinicalTrials.gov format); a line prefix 'I:' or 'E:' sets it per line."""
    crits, ctype = [], default_type
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        h = _HEADER.match(raw)
        if h:
            ctype = h.group(1).lower()
            continue
        t, line = ctype, raw
        p = _PREFIX.match(line)
        if p:
            t = "inclusion" if p.group(1).lower().startswith("i") else "exclusion"
            line = line[p.end():]
        line = _BULLET.sub("", line).strip()
        if len(line) >= 3:
            crits.append({"type": t, "text": line})
    return crits


def criteria_to_text(criteria):
    inc = [c["text"] for c in criteria if c["type"] == "inclusion"]
    exc = [c["text"] for c in criteria if c["type"] == "exclusion"]
    parts = []
    if inc:
        parts += ["Inclusion Criteria:"] + [f"- {t}" for t in inc]
    if exc:
        parts += ([""] if parts else []) + ["Exclusion Criteria:"] + [f"- {t}" for t in exc]
    return "\n".join(parts)


def validate(note, criteria_text):
    if not (note or "").strip():
        raise InputError("Please paste a patient note.")
    if len(note.strip()) < 20:
        raise InputError("The patient note is too short to screen (fewer than 20 characters).")
    sents = split_sentences(note)
    crits = parse_criteria(criteria_text)
    if not crits:
        raise InputError("No criteria found. Enter one criterion per line, under 'Inclusion Criteria:' "
                         "and 'Exclusion Criteria:' headings or with an 'I:' / 'E:' prefix.")
    if len(crits) > MAX_CRITERIA:
        raise InputError(f"{len(crits)} criteria given; the limit is {MAX_CRITERIA} per run.")
    tok, _, _ = MODELS.load_target()
    n_tok = len(tok(data.format_note(sents), add_special_tokens=False)["input_ids"])
    if n_tok > MAX_NOTE_TOKENS:
        raise InputError(f"The note is {n_tok} tokens long; the limit is {MAX_NOTE_TOKENS} "
                         "(training notes were at most ~400 tokens). Please shorten it.")
    return sents, crits


def fetch_ctgov(nct_id):
    """Public trial metadata from ClinicalTrials.gov (API v2). Used only to fill the criteria box."""
    import urllib.request
    nct_id = (nct_id or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise InputError("Enter a ClinicalTrials.gov ID like NCT01397994.")
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields="
           "protocolSection.identificationModule.briefTitle,protocolSection.eligibilityModule.eligibilityCriteria")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            js = json.load(r)
    except Exception as e:
        raise InputError(f"Could not fetch {nct_id} from ClinicalTrials.gov ({e}). Paste the criteria manually.")
    ps = js.get("protocolSection", {})
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    text = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    if not text:
        raise InputError(f"{nct_id} has no eligibility criteria text.")
    return title, criteria_to_text(parse_criteria(text))


# ------------------------------------------------------------------ inference

def _decode(mode, ids, k):
    tok, hf, target = MODELS.load_target()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    if mode == "kv_cache":
        return greedy_cached(target, ids, MAX_NEW_TOKENS, eos)
    if mode == "no_cache":
        return greedy_nocache(target, ids, MAX_NEW_TOKENS, eos)
    if mode == "hf_generate":
        return greedy_hf(hf, ids, MAX_NEW_TOKENS, eos, pad_id=tok.pad_token_id)
    if mode == "speculative":
        return speculative_greedy(target, MODELS.load_draft(), ids, MAX_NEW_TOKENS, k, eos)
    raise ValueError(mode)


def explain(label, evidence, sents, ctype):
    """Template explanation built only from the predicted label and the cited note sentences."""
    by_id = dict(sents)
    quotes = "; ".join(f'[{i}] "{by_id[i]}"' for i in evidence if i in by_id)
    if label is None:
        return "The model output could not be parsed into an allowed label; treat this criterion as unscreened."
    if label == "not enough information":
        return ("The note does not state enough to decide this criterion." +
                (f" Related sentences found: {quotes}, but they do not settle it." if quotes else
                 " No sentence addresses it, and missing information is not treated as evidence either way."))
    if label == "not applicable":
        return "The criterion does not apply to this patient." + (f" Basis: {quotes}." if quotes else "")
    meaning = {"included": "the patient meets this inclusion criterion",
               "not included": "the patient does not meet this inclusion criterion",
               "excluded": "the patient has this excluding condition",
               "not excluded": "the patient does not have this excluding condition"}[label]
    if quotes:
        return f"The model judged that {meaning}, citing {quotes}."
    return (f"The model judged that {meaning}, but cited no supporting sentence, so this decision is "
            "unsupported and should be checked manually.")


def screen(note, criteria_text, mode_label, k=4, reuse_prefix=True, progress=None):
    if progress and not MODELS.ready:
        progress(0, desc="Loading the model (first start downloads about 3 GB)...")
    sents, crits = validate(note, criteria_text)
    mode = MODES[mode_label]
    tok, hf, target = MODELS.load_target()
    valid_ids = [i for i, _ in sents]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    prompts = []
    for c in crits:
        text = tok.apply_chat_template(data.build_messages(sents, c["type"], c["text"]),
                                       tokenize=False, add_generation_prompt=True)
        prompts.append(tok(text, add_special_tokens=False)["input_ids"])

    # Shared-prefix KV reuse: every prompt starts with the same system prompt + patient note,
    # so that prefix is prefilled once and each criterion only prefills its own suffix.
    prefix_len, cache = 0, None
    if mode == "kv_cache" and reuse_prefix and len(prompts) > 1:
        lcp = min(len(p) for p in prompts) - 1
        for p in prompts[1:]:
            lcp = min(lcp, next((i for i, (a, b) in enumerate(zip(prompts[0], p)) if a != b), lcp))
        if lcp > 0:
            prefix_len = lcp
            cache = target.new_cache(max(len(p) for p in prompts) + MAX_NEW_TOKENS)
            if progress:
                progress(0, desc="Encoding the patient note once (shared prefix)...")
            target.forward(torch.tensor([prompts[0][:prefix_len]], device=hf.device), cache)

    # HF-generate mode screens all criteria as left-padded batches (much cheaper per criterion on CPU)
    batched = {}
    if mode == "hf_generate":
        for start in range(0, len(crits), HF_BATCH):
            if progress:
                progress((start, len(crits)), desc=f"Criteria {start + 1}-{min(start + HF_BATCH, len(crits))} "
                                                   f"of {len(crits)} (batched)")
            t0 = time.perf_counter()
            chunk = crits[start:start + HF_BATCH]
            texts = infer.generate_batch(tok, hf, [data.build_messages(sents, c["type"], c["text"]) for c in chunk],
                                         MAX_NEW_TOKENS)
            dt = (time.perf_counter() - t0) / len(chunk)
            for i, t in enumerate(texts):
                batched[start + i] = {"text": t, "n_new": len(tok(t, add_special_tokens=False)["input_ids"]),
                                      "total_s": dt}

    results = []
    t_start = time.perf_counter()
    for n, (c, p_ids) in enumerate(zip(crits, prompts)):
        if progress and n not in batched:
            progress((n, len(crits)), desc=f"Criterion {n + 1} of {len(crits)}")
        ids = torch.tensor([p_ids], device=hf.device)
        if n in batched:
            r = batched[n]
        elif cache is not None:
            r = greedy_cached(target, ids, MAX_NEW_TOKENS, eos, cache=cache, prefix_len=prefix_len)
        else:
            r = _decode(mode, ids, k)
        raw = r["text"] if "text" in r else tok.decode(r["tokens"], skip_special_tokens=True)
        p = data.parse_output(raw, c["type"], valid_ids)
        results.append({
            "criterion_type": c["type"], "criterion": c["text"],
            "label": p["label"] or "unparseable output", "evidence_ids": p["evidence"],
            "evidence_text": [dict(sents)[i] for i in p["evidence"]],
            "explanation": explain(p["label"], p["evidence"], sents, c["type"]),
            "valid_json": p["valid_json"], "raw_output": raw,
            "prompt_tokens": ids.shape[1], "prefill_tokens": ids.shape[1] - prefix_len,
            "new_tokens": r["n_new"], "seconds": round(r["total_s"], 3),
            "acceptance_rate": r.get("acceptance_rate"),
        })
    prov = provenance(note, criteria_text, mode_label, k, time.perf_counter() - t_start)
    prov["shared_prefix_tokens_reused"] = prefix_len
    prov["batched"] = HF_BATCH if mode == "hf_generate" else 1
    return {"sentences": sents, "results": results, "overall": aggregate(results), "provenance": prov}


def aggregate(results):
    """Explicit rule-based patient-trial summary. Not a ground-truth label: the dataset only
    annotates individual criteria."""
    fails = [r for r in results if (r["criterion_type"], r["label"]) in
             {("inclusion", "not included"), ("exclusion", "excluded")}]
    unknown = [r for r in results if r["label"] in ("not enough information", "unparseable output")]
    unsupported = [r for r in results if r["label"] in ("included", "not included", "excluded", "not excluded")
                   and not r["evidence_ids"]]
    if fails:
        verdict = "Likely ineligible (preliminary)"
        reason = f"{len(fails)} criterion/criteria indicate ineligibility."
    elif unknown:
        verdict = "Undetermined: more information needed (preliminary)"
        reason = f"No criterion indicates ineligibility, but {len(unknown)} cannot be decided from the note."
    else:
        verdict = "Potentially eligible (preliminary)"
        reason = "No criterion indicates ineligibility and none lacks information."
    return {"verdict": verdict, "reason": reason,
            "rules": ["any inclusion criterion 'not included' or exclusion criterion 'excluded' -> likely ineligible",
                      "else any 'not enough information' (or unparseable output) -> undetermined",
                      "else -> potentially eligible; 'not applicable' criteria are ignored"],
            "failing": [r["criterion"] for r in fails],
            "missing_information": [r["criterion"] for r in unknown],
            "unsupported_decisions": [r["criterion"] for r in unsupported],
            "counts": {l: sum(r["label"] == l for r in results) for l in sorted({r["label"] for r in results})}}


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("TRIALMATCH_COMMIT", "unknown")


def provenance(note, criteria_text, mode_label, k, seconds):
    dev = MODELS.device()
    return {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target_model": TARGET_BASE, "target_adapter": TARGET_ADAPTER,
            "draft_model": MODELS.draft_name if MODES[mode_label] == "speculative" else None,
            "decoding_mode": mode_label, "draft_length_k": k if MODES[mode_label] == "speculative" else None,
            "decoding": f"greedy, max {MAX_NEW_TOKENS} new tokens, repetition_penalty 1.0",
            "device": torch.cuda.get_device_name(0) if dev is not None and dev.type == "cuda" else "cpu",
            "dtype": str(MODELS.target.dtype).replace("torch.", "") if MODELS.target else None,
            "code_version": _git_commit(), "total_seconds": round(seconds, 2),
            "input_sha256": hashlib.sha256((note + "\n---\n" + criteria_text).encode()).hexdigest()[:16]}


# ------------------------------------------------------------------ report export

def report_html(res, title="Eligibility screening report"):
    e = html.escape
    o, pv = res["overall"], res["provenance"]
    rows = "".join(
        f"<tr><td>{n + 1}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion'])}</td>"
        f"<td><b>{e(r['label'])}</b></td><td>{', '.join(map(str, r['evidence_ids'])) or '—'}</td>"
        f"<td>{'<br>'.join(e(t) for t in r['evidence_text']) or '—'}</td><td>{e(r['explanation'])}</td></tr>"
        for n, r in enumerate(res["results"]))
    note = "".join(f"<li value='{i}'>{e(s)}</li>" for i, s in res["sentences"])
    lst = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs) or "<li>none</li>"
    prov = "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in pv.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}}
.warn{{background:#fff4e5;border:1px solid #f0a020;padding:10px;border-radius:6px}}h2{{margin-top:28px}}</style></head><body>
<h1>{e(title)}</h1><p class="warn"><b>Notice:</b> {e(DISCLAIMER)}</p>
<h2>Preliminary overall outcome (rule-based aggregation)</h2><p><b>{e(o['verdict'])}</b>. {e(o['reason'])}</p>
<p>Rules: {e('; '.join(o['rules']))}.</p>
<h3>Criteria indicating ineligibility</h3><ul>{lst(o['failing'])}</ul>
<h3>Missing information</h3><ul>{lst(o['missing_information'])}</ul>
<h3>Decisions without cited evidence</h3><ul>{lst(o['unsupported_decisions'])}</ul>
<h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th>
<th>Sentence IDs</th><th>Evidence</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note (sentence IDs)</h2><ol start="0">{note}</ol>
<h2>Provenance</h2><table>{prov}</table></body></html>"""


def save_reports(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = os.path.join(out_dir, f"screening_report_{stamp}.html")
    j = os.path.join(out_dir, f"screening_report_{stamp}.json")
    open(h, "w").write(report_html(res))
    json.dump({"notice": DISCLAIMER, **res}, open(j, "w"), indent=2)
    return h, j


# ------------------------------------------------------------------ live inference comparison

def compare_modes(note, criterion, ctype, modes, k):
    """Runs the same prompt through several decoding mechanisms and checks token equality
    against the KV-cache greedy output."""
    sents = split_sentences(note)
    if not sents or not (criterion or "").strip():
        raise InputError("Provide a patient note and one criterion for the comparison.")
    tok, hf, _ = MODELS.load_target()
    msgs = data.build_messages(sents, ctype, criterion.strip())
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=hf.device)
    _decode("kv_cache", ids, k)                               # warm-up
    ref = _decode("kv_cache", ids, k)
    rows = []
    for label in modes:
        r = ref if MODES[label] == "kv_cache" else _decode(MODES[label], ids, k)
        rows.append({"mode": label, "prompt tokens": ids.shape[1], "new tokens": r["n_new"],
                     "prefill s": r["prefill_s"], "decode s": r["decode_s"], "total s": r["total_s"],
                     "tokens/s": r["e2e_tok_s"], "peak extra MiB": r["peak_mem_mb"],
                     "draft acceptance": r.get("acceptance_rate"),
                     "same tokens as KV greedy": first_mismatch(ref["tokens"], r["tokens"]) is None,
                     "output": tok.decode(r["tokens"], skip_special_tokens=True)})
    return rows
