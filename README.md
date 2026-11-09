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
  (602 / 206 / 206 annotations; annotation 883 dropped for empty criterion text). Manifest: [`data/split_manifest.json`](data/split_manifest.json).
  Every script re-derives the split and asserts it equals the manifest.

## Layout
```
data/        split manifest
src/         data.py (loading, split, prompts, parsing) · metrics.py · infer.py · train.py · evaluate.py
             manual_decoder.py (hand-written Qwen2 forward + explicit KV cache) · kv_decode.py · bench_kv.py
             attention.py (reference + tiled online-softmax) · bench_attention.py
             speculative.py (greedy speculative decoding) · bench_spec.py
notebooks/   01 data+split · 02 baseline+fine-tune · 02b oversampling experiment · 03 final test (run once)
             04 KV cache + attention benchmarks · 05 speculative decoding · 06 publish weights + deploy app
results/     predictions, metrics, plots
logs/        training logs, configs, frozen environments
app/         browser application (Gradio): app.py (UI), screening.py (engine), examples.json
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

## Demo video
[5-7 minute application demonstration (YouTube)](https://youtu.be/EhLhlF_ktfA)

## Browser application
Screen a patient note against a trial's criteria (typed, pasted, or fetched from ClinicalTrials.gov by NCT ID),
inspect each criterion's label, cited sentence IDs, evidence text and explanation, see the rule-based preliminary
outcome and missing information, and download an HTML/JSON report with provenance. The *Inference comparison* tab
runs one criterion live through KV-cache, no-cache, speculative and HF-generate decoding and checks token equality.

**Start it (one step):**
```bash
./run_app.sh               # Linux/macOS  (Windows: double-click run_app.bat)
```
The first run creates `.venv`, installs CPU dependencies and downloads ~4 GB of weights; the app then opens at
http://127.0.0.1:7860. Requirements: Python 3.10+, ~8 GB free RAM, ~6 GB disk. Manual alternative:
`pip install -r requirements.txt && python app/app.py`.

Weights (LoRA adapters, public): [target 1.5B](https://huggingface.co/f777777777/trialmatch-qwen2.5-1.5b-lora) ·
[draft 0.5B](https://huggingface.co/f777777777/trialmatch-qwen2.5-0.5b-draft-lora).
The app downloads the LoRA adapters from the Hugging Face Hub. Override with `TRIALMATCH_ADAPTER` /
`TRIALMATCH_DRAFT_ADAPTER` (local folder or Hub id). Runs on CPU (slow, ~10 s per criterion) or GPU.

## Main results (held-out test split, 206 criteria, greedy decoding)
| Model | Accuracy | Macro-F1 | Evidence micro-F1 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct zero-shot | 0.180 | 0.164 | 0.221 |
| + LoRA (r=16) | 0.694 | 0.420 | 0.510 |

KV cache: identical tokens to the uncached path; 5.9x faster at 1024 input / 256 output tokens (T4).
Tiled attention: max error ~1e-6 (fp32), 59x less memory than the reference at 4096 tokens, slower than fused SDPA.
Speculative decoding: 240/240 outputs identical to greedy; 93-98% acceptance with the fine-tuned 0.5B draft,
but 0.8-1.0x speed at batch 1 on a T4 (see report for why).

*Educational screening prototype; not a clinical enrollment decision tool.*
