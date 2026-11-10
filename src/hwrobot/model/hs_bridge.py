"""Bridge to the handwriting-synthesis submodule (third_party/handwriting-synthesis).

That project (a TensorFlow 1 implementation of Graves 2013, by sjvasquez) provides:
- ``drawing.py``: the alphabet and stroke preprocessing, used here directly;
- ``checkpoints/``: the pretrained model, converted once to ``models/hs_weights.npz``
  by ``tools/convert_hs_checkpoint.py`` (needs TensorFlow only for that one step);
- ``styles/``: 13 priming samples (style-N-strokes.npy + style-N-chars.npy).

Its TensorFlow model code can't run on current Python, so :mod:`hwrobot.model.network`
re-implements the same network in PyTorch, layer for layer, so the weights load unchanged.
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def hs_dir() -> Path:
    return Path(os.environ.get("HWROBOT_HS_DIR", REPO_ROOT / "third_party" / "handwriting-synthesis"))


def weights_path() -> Path:
    return Path(os.environ.get("HWROBOT_HS_WEIGHTS", REPO_ROOT / "models" / "hs_weights.npz"))


def available() -> bool:
    return (hs_dir() / "drawing.py").exists() and weights_path().exists()


@lru_cache(maxsize=1)
def drawing():
    """Import the submodule's ``drawing.py`` (matplotlib forced to a headless backend)."""
    path = hs_dir()
    if not (path / "drawing.py").exists():
        raise FileNotFoundError(
            f"{path} is empty. Run: git submodule update --init"
        )
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep the submodule's working tree clean
    try:
        return importlib.import_module("drawing")
    finally:
        sys.dont_write_bytecode = old


def style_ids() -> list[int]:
    styles = hs_dir() / "styles"
    ids = []
    for f in styles.glob("style-*-strokes.npy"):
        try:
            ids.append(int(f.name.split("-")[1]))
        except ValueError:
            pass
    return sorted(ids)


@lru_cache(maxsize=32)
def load_style(style_id: int) -> tuple[np.ndarray, str]:
    """Priming sample: normalized offsets (N, 3) and the text they spell."""
    d = hs_dir() / "styles"
    strokes = np.load(d / f"style-{style_id}-strokes.npy", allow_pickle=False)
    chars = np.load(d / f"style-{style_id}-chars.npy", allow_pickle=False)
    return strokes.astype(np.float32), chars.tobytes().decode("utf-8")
