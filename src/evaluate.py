"""Evaluate the base or fine-tuned model on the val or test split.

  python src/evaluate.py --split val --tag base
  python src/evaluate.py --split val --tag lora --adapter /content/drive/MyDrive/trialmatch/lora_r16/final_adapter
  python src/evaluate.py --split test --tag lora --adapter ... --final-test   # only once, at the very end
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import data
import infer
import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--tag", required=True, help="name used in output files, e.g. base / lora")
    ap.add_argument("--base", default=infer.BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=infer.MAX_NEW_TOKENS)
    ap.add_argument("--final-test", action="store_true",
                    help="required for --split test, so the test set is not touched by accident")
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        sys.exit("Refusing to evaluate on test without --final-test. Tune on val only.")

    rows = data.load_rows()
    manifest = json.load(open(os.path.join(ROOT, "data", "split_manifest.json")))
    fresh = data.make_split(rows)
    assert all(fresh[k] == manifest[k] for k in fresh), "split_manifest.json does not match the seed-42 split"
    data.check_split(manifest, rows)
    split_rows = data.rows_for(rows, manifest, args.split)
    print(f"{args.split}: {len(split_rows)} annotations, model={args.base}, adapter={args.adapter}")

    tok, model = infer.load_model(args.base, args.adapter)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    records = infer.predict_rows(tok, model, split_rows, args.batch_size, args.max_new_tokens)
    elapsed = time.time() - t0

    rep = metrics.full_report(records)
    rep["provenance"] = {
        "split": args.split, "tag": args.tag, "base_model": args.base, "adapter": args.adapter,
        "decoding": {"strategy": "greedy", "do_sample": False, "num_beams": 1,
                     "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens},
        "dtype": str(next(model.parameters()).dtype),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seconds": round(elapsed, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "valid_json_rate": sum(r["valid_json"] for r in records) / len(records),
    }

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    stem = os.path.join(ROOT, "results", f"{args.tag}_{args.split}")
    with open(stem + "_preds.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    json.dump(rep, open(stem + "_metrics.json", "w"), indent=2)

    print(metrics.summary_line(rep))
    for part in ("overall", "inclusion", "exclusion"):
        print(f"\n== {part} ==")
        for lab, c in rep[part]["classification"]["per_class"].items():
            print(f"  {lab:24s} P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f} n={c['support']}")
        metrics.print_confusion(rep[part]["classification"])
    print(f"\nsaved {stem}_preds.jsonl and {stem}_metrics.json")


if __name__ == "__main__":
    main()
