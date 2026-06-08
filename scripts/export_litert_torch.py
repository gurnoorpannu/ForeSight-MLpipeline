"""Export the ForeSight PyTorch checkpoint to Android-stable TFLite.

Run this in Colab or another environment with torch and litert-torch installed.
It bypasses the ONNX/onnx2tf path and preserves the PyTorch context layout:
app_sequences [B, 10], context_sequences [B, 10, 3].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import litert_torch
import torch
import torch.nn as nn


EMBED_DIM = 64
CONTEXT_DIM = 3
LSTM_INPUT_DIM = EMBED_DIM + CONTEXT_DIM
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
VOCAB_SIZE = 87
SEQ_LEN = 10


class NextAppLSTM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.projection = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(
        self,
        app_sequences: torch.Tensor,
        context_sequences: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(app_sequences)
        lstm_input = torch.cat([embedded, context_sequences], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        return self.projection(lstm_out[:, -1, :])


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/foresight_best.pt"),
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/foresight_aet.tflite"),
        help="Path for the LiteRT Torch TFLite export.",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    model = NextAppLSTM(VOCAB_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_recall = checkpoint.get("best_val_recall@3")
    if best_recall is not None:
        print(f"Model loaded. Best val R@3 = {best_recall:.4f}")
    else:
        print("Model loaded.")

    sample_apps = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    sample_context = torch.zeros(1, SEQ_LEN, CONTEXT_DIM, dtype=torch.float32)

    print("Export input contract:")
    print("  app_sequences: [1, 10] int64")
    print("  context_sequences: [1, 10, 3] float32")

    edge_model = litert_torch.convert(model, (sample_apps, sample_context))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(args.output))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Exported: {args.output}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
