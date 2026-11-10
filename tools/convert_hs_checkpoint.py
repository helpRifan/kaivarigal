"""Convert the handwriting-synthesis TF1 checkpoint into arrays our PyTorch model loads.

Needs TensorFlow (any 2.x can read TF1 checkpoints), but only for this one-off step:

    py -3.12 -m venv .tfenv
    .tfenv\\Scripts\\pip install tensorflow-cpu numpy
    .tfenv\\Scripts\\python tools\\convert_hs_checkpoint.py

Reads  third_party/handwriting-synthesis/checkpoints/model-17900
Writes models/hs_weights.npz   (git-ignored)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# TF variable name -> our parameter name
MAPPING = {
    "rnn/LSTMAttentionCell/lstm_cell/kernel": "lstm1.kernel",
    "rnn/LSTMAttentionCell/lstm_cell/bias": "lstm1.bias",
    "rnn/LSTMAttentionCell/attention/weights": "attention.weights",
    "rnn/LSTMAttentionCell/attention/biases": "attention.biases",
    "rnn/LSTMAttentionCell/lstm_cell_1/kernel": "lstm2.kernel",
    "rnn/LSTMAttentionCell/lstm_cell_1/bias": "lstm2.bias",
    "rnn/LSTMAttentionCell/lstm_cell_2/kernel": "lstm3.kernel",
    "rnn/LSTMAttentionCell/lstm_cell_2/bias": "lstm3.bias",
    "rnn/gmm/weights": "gmm.weights",
    "rnn/gmm/biases": "gmm.biases",
}


def main() -> None:
    import tensorflow as tf

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "third_party/handwriting-synthesis/checkpoints/model-17900"))
    ap.add_argument("--out", default=str(ROOT / "models/hs_weights.npz"))
    ap.add_argument("--list", action="store_true", help="only print the checkpoint's variables")
    args = ap.parse_args()

    reader = tf.train.load_checkpoint(args.checkpoint)
    shapes = reader.get_variable_to_shape_map()
    if args.list:
        for name in sorted(shapes):
            print(name, shapes[name])
        return

    missing = [k for k in MAPPING if k not in shapes]
    if missing:
        raise SystemExit(f"checkpoint lacks {missing}; run with --list to see what it has")
    arrays = {ours: reader.get_tensor(theirs).astype(np.float32) for theirs, ours in MAPPING.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    for k, v in arrays.items():
        print(f"{k:20s} {v.shape}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
