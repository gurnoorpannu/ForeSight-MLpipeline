# ForeSight — Predictive On-Device Memory Manager for Android

Proactive memory management system that predicts which app a user will 
open next and acts on the prediction before they tap.

## Model Performance

Trained on LSApp dataset (213K real app switches across 291 users,  
after collapsing consecutive duplicate events).

| Metric | Score | Baseline (most frequent in window) |
|--------|-------|-------------------------------------|
| Recall@1 | 37.8% | 9.5% |
| Recall@3 | 59.7% | 25.9% |
| Recall@5 | 67.8% | — |
| Recall@8 | 74.4% | — |

## Architecture

- 2-layer LSTM, 64-dim app embeddings + 3-dim context features
- Input: last 10 app switches + (hour of day, day of week, time gap)
- Output: probability distribution over 87 apps
- Parameters: ~250K — current ONNX-converted TFLite fp32: **0.96 MB**

## Key Engineering Notes

- Caught and fixed data leakage bug: LSApp logs multiple OPENED events 
  per Activity, causing 83.5% baseline from simply copying last event.
  After deduplication: 1.67M raw events → 213K real app switches.
- onnx2tf transposes context axis: PyTorch `[B, 10, 3]` → TFLite `[B, 3, 10]`
- The ONNX/onnx2tf TFLite export loads on Android but crashes during native
  `Interpreter.runForMultipleInputsOutputs`; re-export with ai-edge-torch for
  the Android production artifact.
- ai-edge-torch preserves PyTorch context layout: `[B, 10, 3]`
- fp16 quantization incompatible with TFLite GATHER op (embedding layer)

## Android TFLite Export

Use the PyTorch checkpoint as the source of truth and export directly with
Google's ai-edge-torch path:

```bash
python scripts/export_ai_edge_torch.py \
  --checkpoint models/foresight_best.pt \
  --output models/foresight_aet.tflite
```

In Colab, point `--checkpoint` and `--output` at the Drive model directory.
The generated `foresight_aet.tflite` should replace the old ONNX-converted
asset in the Android app after verifying that Android LiteRT can invoke it
without a native crash.

Android input contract for the ai-edge-torch export:

| Input | Shape | Type |
|-------|-------|------|
| `app_sequences` | `[1, 10]` | `int64` |
| `context_sequences` | `[1, 10, 3]` | `float32` |

## Files

| File | Description |
|------|-------------|
| `models/foresight_fp32.tflite` | ONNX/onnx2tf export; not Android-stable |
| `models/foresight_best.pt` | PyTorch checkpoint |
| `models/foresight_lstm.onnx` | ONNX intermediate |
| `outputs/app_vocab.json` | App ID → package name mapping |
| `notebook/ForeSightMLPipeline.ipynb` | Full reproducible pipeline |
| `scripts/export_ai_edge_torch.py` | Direct PyTorch → TFLite export script |
