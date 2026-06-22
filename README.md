# ForeSight ML Pipeline

Training and export pipeline for **ForeSight**, a predictive on-device memory manager for Android.

This repo trains the next-app-prediction LSTM and exports it to TFLite. The Android app that runs it on-device lives at [ForeSightApk](https://github.com/gurnoorpannu/ForeSightApk).

## What it predicts

Given the last 10 app switches plus light context (hour of day, day of week, time since last switch), the model predicts which app you'll open next, out of an 87-app vocabulary built from the training data.

## Model performance

Trained on the LSApp dataset (213K real app switches across 291 users, after deduplicating consecutive identical events):

| Metric | Score | Baseline (most-frequent-in-window) |
|---|---|---|
| Recall@1 | 37.8% | 9.5% |
| Recall@3 | 59.7% | 25.9% |
| Recall@5 | 67.8% | — |
| Recall@8 | 74.4% | — |

## Architecture

- 2-layer LSTM (hidden size 128, dropout 0.3), 64-dim app embeddings concatenated with 3-dim context features at every timestep
- Input: 10-step app sequence `[B,10]` (int64 app IDs) + context `[B,10,3]` (hour-of-day, day-of-week, time-gap — all normalized)
- Output: raw logits over 87 app classes
- ~250K parameters — exported size 1.01 MB (fp32, LiteRT Torch)

## Key engineering notes

- **Data leakage bug.** LSApp logs multiple `OPENED` events per Android Activity, not per real app switch. Treated naively, this let the model hit 83.5% "accuracy" by trivially copying the last event. Deduplicating consecutive repeats brought the dataset down from 1.67M raw events to 213K genuine app switches, and real recall dropped to the numbers above.
- **ONNX export crashed on Android.** The original `onnx2tf` path silently transposed the context axis (`[B,10,3]` → `[B,3,10]`) and produced a graph that loaded fine on-device but segfaulted (`SIGSEGV`) inside `Interpreter.runForMultipleInputsOutputs` — even after ruling out tensor routing, XNNPACK, and 16KB page alignment as causes.
- **Fix: export straight from PyTorch via LiteRT Torch** (`litert_torch.convert`), bypassing ONNX entirely. This preserves the original `[B,10,3]` context layout and produces a graph that runs cleanly on-device.
- fp16 quantization is incompatible with the TFLite `GATHER` op used by the embedding layer, so the shipped model is fp32.

## Android contract

The exported model expects:

| Input | Shape | Type |
|---|---|---|
| `app_sequences` | `[1, 10]` | `int64` |
| `context_sequences` | `[1, 10, 3]` | `float32` |

Output: `[1, 87]` float32 raw logits (softmax applied on-device).

## Repo layout

```
models/
├── foresight_best.pt        # PyTorch checkpoint (source of truth)
├── foresight_lstm.onnx      # ONNX intermediate (legacy export path, kept for reference)
└── foresight_aet.tflite     # Android production model — LiteRT Torch export, 1.01 MB

notebook/
├── ForeSightMLPipeline.ipynb         # Original end-to-end pipeline: data prep -> train -> eval
└── ForeSightMLPipeline_LiteRT.ipynb  # Latest pipeline, includes LiteRT Torch export

outputs/
├── app_vocab.json           # App label -> model app-ID mapping (87 entries)
├── class_distribution.csv   # Per-app target counts in the training set
└── metadata.json            # Tensor schema, sequence length, target event type, seed

scripts/
└── export_litert_torch.py   # Standalone PyTorch checkpoint -> TFLite (LiteRT Torch) exporter
```

## Reproducing the export

```bash
pip install torch litert-torch
python scripts/export_litert_torch.py \
  --checkpoint models/foresight_best.pt \
  --output models/foresight_aet.tflite
```

For the full data pipeline (loading LSApp, deduplication, training, evaluation), see `notebook/ForeSightMLPipeline_LiteRT.ipynb`.

## Related

- Android app (runs this model on-device): [ForeSightApk](https://github.com/gurnoorpannu/ForeSightApk)
