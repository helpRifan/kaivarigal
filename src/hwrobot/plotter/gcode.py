"""Strokes -> G-code (Phase 5).

Will convert absolute-mm points (see :mod:`hwrobot.strokes`) into GRBL G-code:
servo pen up/down commands, G0 travel moves, G1 drawing moves at a feed rate.
Note: plotter Y axis may point the other way from page +y-down; the flip
belongs here (or in calibration), not in the stroke format.
"""

from __future__ import annotations

import numpy as np


def strokes_to_gcode(points: np.ndarray, feed_rate_mm_min: float = 1500.0) -> list[str]:
    """Convert absolute-mm points to G-code lines. Not implemented yet."""
    raise NotImplementedError("G-code generation is Phase 5")
