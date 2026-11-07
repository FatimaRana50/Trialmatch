# TrialMatch: criterion-level clinical-trial eligibility with a fine-tuned open Transformer

Educational screening prototype. **Not a clinical enrollment decision tool.**

Given a patient note, a criterion type (inclusion/exclusion) and the criterion text, the model returns
`{"evidence": [sentence IDs], "label": "..."}` with the dataset's own label set:

| criterion type | labels |
|---|---|
| inclusion | included · not included · not enough information · not applicable |
| exclusion | excluded · not excluded · not enough information · not applicable |

## Data and split
- Dataset: [`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (1015 annotations, 53 patients, 103 trials).
- Columns used: see `KEEP_COLUMNS` in `src/data.py`. All `gpt4_*` columns, `explanation_correctness` and `training` are dropped.
- Split: sorted patient IDs → `np.random.default_rng(42).shuffle` → 37 train / 8 val / 8 test patients
  (603 / 206 / 206 annotations). Manifest: [`data/split_manifest.json`](data/split_manifest.json).
  Every script re-derives the split and asserts it equals the manifest.

## Layout
```
data/        split manifest
src/         data.py (loading, split, prompts, parsing) · metrics.py · infer.py · train.py · evaluate.py
notebooks/   01 data+split · 02 baseline+fine-tune · 03 final test (run once)
results/     predictions, metrics, plots
logs/        training logs, configs, frozen environments
app/         browser application (to come)
```

## Reproduce (Google Colab, free T4)
1. Add a Colab secret `GH_TOKEN` (GitHub personal access token with repo access).
2. Run `notebooks/01_data_and_split.ipynb`, then `02_baseline_and_finetune.ipynb`.
3. After all tuning is frozen, run `03_final_test.ipynb` once.

Command-line equivalents:
```bash
pip install -r requirements.txt
python src/evaluate.py --split val --tag base
python src/train.py --out checkpoints/lora_r16
python src/evaluate.py --split val --tag lora --adapter checkpoints/lora_r16/final_adapter
python src/evaluate.py --split test --tag lora --adapter checkpoints/lora_r16/final_adapter --final-test
```
