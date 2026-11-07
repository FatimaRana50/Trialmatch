"""Criterion classification and evidence-retrieval metrics (pure Python)."""
from collections import Counter

INVALID = "<invalid>"


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification(gold, pred, labels=None):
    """gold/pred: lists of label strings (pred may be None -> counted as INVALID, always wrong).
    Macro-F1 averages over `labels`; by default the labels that occur in gold, so a class
    with no test support does not add a forced 0."""
    pred = [p if p is not None else INVALID for p in pred]
    if labels is None:
        labels = sorted(set(gold))
    per_class = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        p, r, f = _prf(tp, fp, fn)
        per_class[l] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    cm_labels = list(labels) + sorted({p for p in pred if p not in labels})
    cm = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in cm_labels] for a in labels]
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0,
        "per_class": per_class,
        "confusion": {"rows_gold": list(labels), "cols_pred": cm_labels, "matrix": cm},
        "invalid_outputs": sum(p == INVALID for p in pred),
    }


def evidence(gold_sets, pred_sets):
    """Sentence-ID retrieval.
    micro: TP/FP/FN pooled over every record (empty gold records can only add FPs).
    macro: mean of per-record P/R/F1 over records whose gold set is non-empty
           (an empty prediction there scores P=R=F1=0).
    empty_gold_*: on records with no annotated evidence, how often the model also cites nothing."""
    tp = fp = fn = 0
    mp, mr, mf = [], [], []
    empty_total = empty_ok = 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if g:
            pr, rc, f = _prf(len(g & p), len(p - g), len(g - p))
            mp.append(pr); mr.append(rc); mf.append(f)
        else:
            empty_total += 1
            empty_ok += not p
    P, R, F = _prf(tp, fp, fn)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "micro_precision": P, "micro_recall": R, "micro_f1": F,
        "macro_precision": avg(mp), "macro_recall": avg(mr), "macro_f1": avg(mf),
        "records_with_gold": len(mf),
        "empty_gold_records": empty_total,
        "empty_gold_correctly_empty": empty_ok / empty_total if empty_total else 0.0,
    }


def full_report(records):
    """records: dicts with criterion_type, gold_label, pred_label, gold_evidence, pred_evidence.
    Returns overall + inclusion-only + exclusion-only results."""
    out = {}
    for name, sub in [("overall", records),
                      ("inclusion", [r for r in records if r["criterion_type"] == "inclusion"]),
                      ("exclusion", [r for r in records if r["criterion_type"] == "exclusion"])]:
        out[name] = {
            "classification": classification([r["gold_label"] for r in sub],
                                             [r["pred_label"] for r in sub]),
            "evidence": evidence([r["gold_evidence"] for r in sub],
                                 [r["pred_evidence"] for r in sub]),
        }
    return out


def summary_line(rep):
    o = rep["overall"]
    return (f"acc={o['classification']['accuracy']:.3f}  macroF1={o['classification']['macro_f1']:.3f}  "
            f"incl_acc={rep['inclusion']['classification']['accuracy']:.3f}  "
            f"excl_acc={rep['exclusion']['classification']['accuracy']:.3f}  "
            f"evid_microF1={o['evidence']['micro_f1']:.3f}  evid_macroF1={o['evidence']['macro_f1']:.3f}  "
            f"invalid={o['classification']['invalid_outputs']}")


def print_confusion(cls):
    c = cls["confusion"]
    w = max(len(x) for x in c["cols_pred"] + c["rows_gold"]) + 2
    print("gold \\ pred".ljust(w) + "".join(x[:12].rjust(14) for x in c["cols_pred"]))
    for lab, row in zip(c["rows_gold"], c["matrix"]):
        print(lab.ljust(w) + "".join(str(v).rjust(14) for v in row))
