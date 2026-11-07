"""Loading, fixed patient-grouped split, and prompt/target formatting for
the TrialGPT Criterion Annotations dataset."""
import json
import re

import numpy as np

DATASET = "ncbi/TrialGPT-Criterion-Annotations"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

# Only these columns are ever used. Everything else is GPT-4 output
# (gpt4_explanation, gpt4_sentences, gpt4_eligibility), a grade of GPT-4 output
# (explanation_correctness) or TrialGPT's own split flag (training).
KEEP_COLUMNS = ["annotation_id", "patient_id", "note", "trial_id", "trial_title",
                "criterion_type", "criterion_text", "expert_sentences", "expert_eligibility"]

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]

LABEL_HELP = {
    "inclusion": (
        '- "included": the note shows the patient meets the criterion.\n'
        '- "not included": the note shows the patient does not meet the criterion.\n'
        '- "not enough information": the note does not say enough to decide.\n'
        '- "not applicable": the criterion does not apply to this patient.'),
    "exclusion": (
        '- "excluded": the note shows the patient has the excluding condition.\n'
        '- "not excluded": the note shows the patient does not have the excluding condition.\n'
        '- "not enough information": the note does not say enough to decide.\n'
        '- "not applicable": the criterion does not apply to this patient.'),
}

SYSTEM_PROMPT = (
    "You are a clinical trial screening assistant. You judge one eligibility criterion "
    "against one patient note, using only what the note says. Absence of a statement is "
    "not proof either way. Reply with JSON only.")

_SENT_RE = re.compile(r"^(\d+)\.\s?(.*)$")


def parse_ids(s):
    """'[0, 3]' -> [0, 3]; '' / None / '[]' -> []."""
    if s is None or str(s).strip() == "":
        return []
    return sorted({int(i) for i in json.loads(s)})


def split_note(note):
    """The dataset note is already one numbered sentence per line ('0. ...').
    Returns a list of (id, text). Lines without a number are appended to the
    previous sentence so no text is lost."""
    sents = []
    for line in note.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SENT_RE.match(line)
        if m:
            sents.append((int(m.group(1)), m.group(2).strip()))
        elif sents:
            sid, txt = sents[-1]
            sents[-1] = (sid, txt + " " + line)
        else:
            sents.append((0, line))
    return sents


def clean_row(r):
    out = {k: r[k] for k in KEEP_COLUMNS}
    out["expert_sentences"] = parse_ids(r["expert_sentences"])
    out["sentences"] = split_note(r["note"])
    return out


def is_usable(r):
    """Annotation 883 has criterion_text=None; with no criterion there is nothing to judge."""
    return bool((r["criterion_text"] or "").strip()) and bool((r["note"] or "").strip())


def load_rows():
    """Returns cleaned usable rows. Unusable annotation IDs are stored in DROPPED."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    DROPPED[:] = [r["annotation_id"] for r in ds if not is_usable(r)]
    if DROPPED:
        print(f"dropped {len(DROPPED)} annotation(s) with empty criterion/note: {DROPPED}")
    return [clean_row(r) for r in ds if is_usable(r)]


DROPPED = []


def make_split(rows):
    """Sort unique patient IDs, shuffle with default_rng(42), take 37/8/8."""
    pids = np.array(sorted({r["patient_id"] for r in rows}))
    assert len(pids) == N_TRAIN + N_VAL + N_TEST, len(pids)
    np.random.default_rng(SEED).shuffle(pids)
    pids = [str(p) for p in pids]
    return {"train": pids[:N_TRAIN],
            "val": pids[N_TRAIN:N_TRAIN + N_VAL],
            "test": pids[N_TRAIN + N_VAL:]}


def check_split(split, rows):
    """Raise if any patient is in two splits or missing from all of them."""
    sets = {k: set(split[k]) for k in ("train", "val", "test")}
    for a in sets:
        for b in sets:
            if a < b:
                assert not sets[a] & sets[b], f"patient leakage between {a} and {b}"
    all_pids = {r["patient_id"] for r in rows}
    assert set().union(*sets.values()) == all_pids, "split does not cover all patients"
    # each annotation lands in exactly one split
    for r in rows:
        assert sum(r["patient_id"] in s for s in sets.values()) == 1


def build_manifest(rows):
    split = make_split(rows)
    check_split(split, rows)
    manifest = {
        "dataset": DATASET,
        "procedure": "sorted unique patient_id -> np.random.default_rng(42).shuffle -> first 37 train, next 8 val, last 8 test",
        "seed": SEED,
        **split,
        "dropped_annotations": {"ids": list(DROPPED), "reason": "empty criterion_text"},
        "counts": {},
    }
    for k in ("train", "val", "test"):
        sub = [r for r in rows if r["patient_id"] in set(split[k])]
        manifest["counts"][k] = {
            "patients": len(split[k]),
            "annotations": len(sub),
            "inclusion": sum(r["criterion_type"] == "inclusion" for r in sub),
            "exclusion": sum(r["criterion_type"] == "exclusion" for r in sub),
            "labels": {l: sum(r["expert_eligibility"] == l for r in sub) for l in ALL_LABELS},
            "empty_evidence": sum(not r["expert_sentences"] for r in sub),
            "annotation_ids": sorted(r["annotation_id"] for r in sub),
        }
    return manifest


def rows_for(rows, manifest, split_name):
    keep = set(manifest[split_name])
    return [r for r in rows if r["patient_id"] in keep]


# ---------------------------------------------------------------- prompts

def format_note(sentences):
    return "\n".join(f"{i}. {t}" for i, t in sentences)


def user_prompt(sentences, criterion_type, criterion_text):
    labels = ", ".join(f'"{l}"' for l in LABELS[criterion_type])
    return (
        f"Patient note (each sentence starts with its ID):\n{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed labels: {labels}\n{LABEL_HELP[criterion_type]}\n\n"
        'Return JSON: {"evidence": [IDs of the note sentences that support the decision], '
        '"label": "<one allowed label>"}. Use an empty evidence list if no sentence is relevant.')


def build_messages(sentences, criterion_type, criterion_text):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(sentences, criterion_type, criterion_text)}]


def target_text(row):
    # evidence comes first so the label is generated after the cited sentences
    return json.dumps({"evidence": row["expert_sentences"], "label": row["expert_eligibility"]})


def row_messages(row, with_target=False):
    msgs = build_messages(row["sentences"], row["criterion_type"], row["criterion_text"])
    if with_target:
        msgs.append({"role": "assistant", "content": target_text(row)})
    return msgs


# ---------------------------------------------------------------- parsing

def parse_output(text, criterion_type, valid_ids):
    """Model text -> dict(label, evidence, valid). Label is None if nothing usable."""
    allowed = LABELS[criterion_type]
    valid_ids = set(valid_ids)
    label, evidence, ok = None, [], False
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).strip().lower()
            if lab in allowed:
                label, ok = lab, True
            ev = obj.get("evidence", [])
            if isinstance(ev, list):
                evidence = sorted({int(i) for i in ev
                                   if str(i).lstrip("-").isdigit() and int(i) in valid_ids})
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass
    if label is None:
        # fallback: longest allowed label mentioned in the text
        # (longest first so "not included" wins over "included")
        low = text.lower()
        for lab in sorted(allowed, key=len, reverse=True):
            if f'"{lab}"' in low or lab in low:
                label = lab
                break
    return {"label": label, "evidence": evidence, "valid_json": ok}
