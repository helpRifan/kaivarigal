"""Handwriting generation model (Phase 2).

Graves-style LSTM + soft attention window over characters + mixture density
output (arXiv:1308.0850). Pretrain on IAM-OnDB, fine-tune on my samples.
Works in normalized relative offsets (dx, dy, pen_up); see
:func:`hwrobot.strokes.normalize_offsets`. Controls: bias ("neatness"),
slant, size. Must stay small enough for CPU inference on a Pi 4.

Planned modules: ``dataset.py``, ``network.py``, ``train.py``, ``sample.py``.
"""
