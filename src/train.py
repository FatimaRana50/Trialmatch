"""LoRA supervised fine-tuning of Qwen2.5-1.5B-Instruct on the training split.
Loss is computed on the assistant JSON only (prompt tokens are masked with -100).
Validation loss is used for checkpoint selection; the test split is never loaded.

  python src/train.py --out /content/drive/MyDrive/trialmatch/lora_r16
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

import data
import infer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode(tok, row, max_len):
    prompt = tok.apply_chat_template(data.row_messages(row), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(data.row_messages(row, with_target=True), tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "chat template prompt is not a prefix of the full text"
    if len(f_ids) > max_len:
        # never silently cut evidence: fail loudly (notebook 01 shows max length is far below this)
        raise ValueError(f"annotation {row['annotation_id']} has {len(f_ids)} tokens > max_len={max_len}")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def oversample(rows, target, max_repeat):
    """Repeat every row of a rare label round(target / count) times (capped), deterministically.
    Only the training split is passed in; val is untouched."""
    counts = Counter(r["expert_eligibility"] for r in rows)
    factor = {l: max(1, min(max_repeat, round(target / c))) for l, c in counts.items()}
    out = [r for r in rows for _ in range(factor[r["expert_eligibility"]])]
    print("oversampling factors:", factor)
    print("label counts after:", dict(Counter(r["expert_eligibility"] for r in out)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=infer.BASE_MODEL)
    ap.add_argument("--out", required=True, help="checkpoint dir (put it on Google Drive)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true", help="load base in fp32 (use if fp16 loss becomes NaN)")
    ap.add_argument("--oversample", type=int, default=0,
                    help="repeat rare-label training rows until each label has about this many (0 = off)")
    ap.add_argument("--max-repeat", type=int, default=6, help="cap on how often one row is repeated")
    ap.add_argument("--tag", default="", help="suffix for log/plot file names, e.g. os")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    set_seed(args.seed)

    rows = data.load_rows()
    manifest = json.load(open(os.path.join(ROOT, "data", "split_manifest.json")))
    data.check_split(manifest, rows)
    train_rows = data.rows_for(rows, manifest, "train")
    val_rows = data.rows_for(rows, manifest, "val")
    if args.oversample:
        train_rows = oversample(train_rows, args.oversample, args.max_repeat)

    tok = AutoTokenizer.from_pretrained(args.base)
    train_ds = [encode(tok, r, args.max_len) for r in train_rows]
    val_ds = [encode(tok, r, args.max_len) for r in val_rows]
    lens = [len(x["input_ids"]) for x in train_ds]
    print(f"train={len(train_ds)} val={len(val_ds)} tokens: max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    dtype = torch.float32 if args.fp32 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for gradient checkpointing with a frozen base
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)  # LoRA weights are kept in fp32
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc, lr_scheduler_type="cosine", warmup_steps=10,
        weight_decay=0.0, logging_steps=2, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=not args.fp32, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.out, "final_adapter")
    trainer.save_model(final)  # best checkpoint by val loss
    tok.save_pretrained(final)
    print("best checkpoint:", trainer.state.best_model_checkpoint, "->", final)

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    json.dump(trainer.state.log_history, open(os.path.join(ROOT, "logs", f"train_log{sfx}.json"), "w"), indent=2)
    json.dump({**vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_val_loss": trainer.state.best_metric,
               "train_examples": len(train_ds), "val_examples": len(val_ds),
               "max_train_tokens": max(lens),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None},
              open(os.path.join(ROOT, "logs", f"train_config{sfx}.json"), "w"), indent=2)
    plot_curves(trainer.state.log_history, os.path.join(ROOT, "results", f"training_curves{sfx}.png"))


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*tr), label="train loss")
    plt.plot(*zip(*ev), marker="o", label="val loss")
    plt.xlabel("optimizer step"); plt.ylabel("cross-entropy (response tokens)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
