"""Model loading and batched greedy prediction (used by evaluation and the app)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import data

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 64


def load_model(base=BASE_MODEL, adapter=None, dtype=torch.float16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"  # left-pad so every prompt ends right before generation
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.to(device).eval()


def greedy_config(tok, max_new_tokens=MAX_NEW_TOKENS):
    # Qwen's shipped generation_config samples (temperature 0.7, repetition_penalty 1.05).
    # Replace it fully so decoding is deterministic greedy.
    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])


@torch.no_grad()
def generate_batch(tok, model, messages_list, max_new_tokens=MAX_NEW_TOKENS):
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
               for m in messages_list]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, generation_config=greedy_config(tok, max_new_tokens))
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def predict_rows(tok, model, rows, batch_size=16, max_new_tokens=MAX_NEW_TOKENS, progress=True):
    # sort by prompt length so batches need little padding; restore order afterwards
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["note"]) + len(rows[i]["criterion_text"]))
    outputs = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        texts = generate_batch(tok, model, [data.row_messages(rows[i]) for i in idx], max_new_tokens)
        for i, t in zip(idx, texts):
            outputs[i] = t
        if progress:
            print(f"\r{min(start + batch_size, len(rows))}/{len(rows)}", end="", flush=True)
    if progress:
        print()
    records = []
    for r, raw in zip(rows, outputs):
        parsed = data.parse_output(raw, r["criterion_type"], [i for i, _ in r["sentences"]])
        records.append({
            "annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
            "trial_id": r["trial_id"], "criterion_type": r["criterion_type"],
            "criterion_text": r["criterion_text"],
            "gold_label": r["expert_eligibility"], "gold_evidence": r["expert_sentences"],
            "pred_label": parsed["label"], "pred_evidence": parsed["evidence"],
            "valid_json": parsed["valid_json"], "raw_output": raw,
        })
    return records
