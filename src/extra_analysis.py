"""Extra analyses for the report: per-split statistics, cross-split leakage checks,
patient-level bootstrap CIs, fixed/broken counts, error taxonomy and unsupported-decision rates.

  python src/extra_analysis.py      -> results/extra_analysis.json
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from transformers import AutoTokenizer

import data
import infer
import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIVE = {"included", "not included", "excluded", "not excluded"}


def error_type(gold, pred):
    if gold == pred:
        return None
    if gold in DECISIVE and pred in DECISIVE:
        return "polarity"
    if gold == "not enough information" and pred in DECISIVE:
        return "decision_without_info"
    if "not applicable" in (gold, pred):
        return "applicability"
    if gold in DECISIVE and pred == "not enough information":
        return "missed_info"
    return "other"


def preds(tag, split):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", f"{tag}_{split}_preds.jsonl"))]


def bootstrap(base, lora, n=2000, seed=0):
    """Resample whole patients (annotations of one patient are correlated)."""
    pids = sorted({r["patient_id"] for r in base})
    idx = {p: [i for i, r in enumerate(base) if r["patient_id"] == p] for p in pids}
    rng = np.random.default_rng(seed)
    acc_gain, f1_gain, lora_acc = [], [], []
    for _ in range(n):
        s = sum((idx[p] for p in rng.choice(pids, len(pids))), [])
        cb = metrics.classification([base[i]["gold_label"] for i in s], [base[i]["pred_label"] for i in s])
        cl = metrics.classification([lora[i]["gold_label"] for i in s], [lora[i]["pred_label"] for i in s])
        acc_gain.append(cl["accuracy"] - cb["accuracy"])
        f1_gain.append(cl["macro_f1"] - cb["macro_f1"])
        lora_acc.append(cl["accuracy"])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"acc_gain": round(float(np.mean(acc_gain)), 3), "acc_gain_ci": ci(acc_gain),
            "mf1_gain": round(float(np.mean(f1_gain)), 3), "mf1_gain_ci": ci(f1_gain), "lora_acc_ci": ci(lora_acc)}


def main():
    rows = data.load_rows()
    manifest = json.load(open(os.path.join(ROOT, "data", "split_manifest.json")))
    tok = AutoTokenizer.from_pretrained(infer.BASE_MODEL)
    out = {}
    splits = ("train", "val", "test")
    for sp in splits:
        R = data.rows_for(rows, manifest, sp)
        lens = [len(tok(tok.apply_chat_template(data.row_messages(r), tokenize=False, add_generation_prompt=True),
                        add_special_tokens=False)["input_ids"]) for r in R]
        out[sp] = {"patients": len(manifest[sp]), "trials": len({r["trial_id"] for r in R}), "n": len(R),
                   "incl": sum(r["criterion_type"] == "inclusion" for r in R),
                   "excl": sum(r["criterion_type"] == "exclusion" for r in R),
                   "labels": {l: sum(r["expert_eligibility"] == l for r in R) for l in data.ALL_LABELS},
                   "empty": sum(not r["expert_sentences"] for r in R),
                   "tok_mean": round(float(np.mean(lens))), "tok_max": max(lens)}
    notes = {sp: {r["note"] for r in data.rows_for(rows, manifest, sp)} for sp in splits}
    pairs = {sp: {(r["note"], r["criterion_text"]) for r in data.rows_for(rows, manifest, sp)} for sp in splits}
    combos = [("train", "val"), ("train", "test"), ("val", "test")]
    out["leak"] = {"notes": [len(notes[a] & notes[b]) for a, b in combos],
                   "pairs": [len(pairs[a] & pairs[b]) for a, b in combos]}

    for sp in ("val", "test"):
        b, l = preds("base", sp), preds("lora", sp)
        assert [r["annotation_id"] for r in b] == [r["annotation_id"] for r in l]
        out[sp + "_boot"] = {
            "fixed": sum(x["pred_label"] != x["gold_label"] and y["pred_label"] == y["gold_label"] for x, y in zip(b, l)),
            "broken": sum(x["pred_label"] == x["gold_label"] and y["pred_label"] != y["gold_label"] for x, y in zip(b, l)),
            **bootstrap(b, l)}
        out[sp + "_tax"] = {t: dict(collections.Counter(error_type(r["gold_label"], r["pred_label"]) for r in R
                                                        if r["gold_label"] != r["pred_label"]))
                            for t, R in (("base", b), ("lora", l))}
        # [decisive with no cited sentence, decisive total, of which correct]
        out[sp + "_unsupported"] = {t: [sum(r["pred_label"] in DECISIVE and not r["pred_evidence"] for r in R),
                                        sum(r["pred_label"] in DECISIVE for r in R),
                                        sum(r["pred_label"] in DECISIVE and not r["pred_evidence"]
                                            and r["pred_label"] == r["gold_label"] for r in R)]
                                    for t, R in (("base", b), ("lora", l))}
        out[sp + "_expert_unsup"] = [sum(r["gold_label"] in DECISIVE and not r["gold_evidence"] for r in l),
                                     sum(r["gold_label"] in DECISIVE for r in l)]
    json.dump(out, open(os.path.join(ROOT, "results", "extra_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
