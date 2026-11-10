"""Shared stroke format — the contract between every module.

There are exactly two representations, both ``float`` NumPy arrays of shape ``(N, 3)``:

**Points** (absolute) — ``(x, y, pen_up)``
    Used by capture, layout, plotter, everything after the model.
    Units are millimetres on the page, origin top-left, +y down.
    ``pen_up == 1`` means *the stroke ends after this point* (lift the pen
    before moving to the next point). The last point always has ``pen_up == 1``.

**Offsets** (relative) — ``(dx, dy, pen_up)``
    Used by the model, IAM-OnDB style. Row ``i`` is the move from point
    ``i-1`` to point ``i``; row 0 is relative to the origin ``(0, 0)``, so
    ``offsets_to_points(points_to_offsets(p)) == p`` exactly.
    ``pen_up`` has the same meaning as above. Offsets are usually
    *normalized* (divided by a dataset-wide scale) before training; see
    :func:`normalize_offsets`.

This module owns all conversions. Other modules must not invent their own format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: ``(N, 3)`` float array of ``(x, y, pen_up)`` in absolute mm.
Points = np.ndarray
#: ``(N, 3)`` float array of ``(dx, dy, pen_up)``.
Offsets = np.ndarray


# --------------------------------------------------------------------------- #
# Validation / construction
# --------------------------------------------------------------------------- #


def as_array(data: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Coerce ``data`` to a ``(N, 3)`` float64 array and validate it.

    Checks: shape ``(N, 3)``, all values finite, ``pen_up`` is 0 or 1, and
    (if non-empty) the final row has ``pen_up == 1``. Works for both points and
    offsets. Returns a new array; the input is never modified.
    """
    arr = np.array(data, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"stroke array must have shape (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("stroke array contains NaN or inf")
    if not np.all(np.isin(arr[:, 2], (0.0, 1.0))):
        raise ValueError("pen_up column must contain only 0 or 1")
    if arr[-1, 2] != 1.0:
        raise ValueError("last point must have pen_up == 1 (every stroke must end)")
    return arr


def from_strokes(strokes: Iterable[Sequence[Sequence[float]]]) -> Points:
    """Build points from a list of strokes, each a list of ``(x, y)`` pairs.

    Empty strokes are skipped. The last point of each stroke gets ``pen_up = 1``.

    >>> from_strokes([[(0, 0), (1, 0)], [(2, 2)]]).tolist()
    [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 2.0, 1.0]]
    """
    rows: list[np.ndarray] = []
    for stroke in strokes:
        xy = np.array(stroke, dtype=np.float64).reshape(-1, 2)
        if len(xy) == 0:
            continue
        pen = np.zeros((len(xy), 1))
        pen[-1, 0] = 1.0
        rows.append(np.hstack([xy, pen]))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return as_array(np.vstack(rows))


def split_strokes(points: Points) -> list[np.ndarray]:
    """Split points into a list of ``(M, 2)`` xy arrays, one per pen-down stroke."""
    points = as_array(points)
    if len(points) == 0:
        return []
    ends = np.flatnonzero(points[:, 2] == 1.0) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return [points[s:e, :2].copy() for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------- #
# Absolute <-> relative
# --------------------------------------------------------------------------- #


def points_to_offsets(points: Points) -> Offsets:
    """Absolute points -> relative offsets (row 0 is relative to the origin)."""
    points = as_array(points)
    offsets = points.copy()
    offsets[1:, :2] = np.diff(points[:, :2], axis=0)
    return offsets


def offsets_to_points(offsets: Offsets, origin: tuple[float, float] = (0.0, 0.0)) -> Points:
    """Relative offsets -> absolute points, starting from ``origin``."""
    offsets = as_array(offsets)
    points = offsets.copy()
    points[:, :2] = np.cumsum(offsets[:, :2], axis=0) + np.asarray(origin, dtype=np.float64)
    return points


# --------------------------------------------------------------------------- #
# Normalization (model side)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NormStats:
    """Scale used to normalize offsets. Saved alongside a model checkpoint."""

    scale: float

    @classmethod
    def fit(cls, offset_sets: Iterable[Offsets]) -> "NormStats":
        """Compute the std of all ``(dx, dy)`` values across a dataset."""
        xy = [as_array(o)[:, :2] for o in offset_sets]
        xy = [a for a in xy if len(a)]
        if not xy:
            raise ValueError("cannot fit NormStats on empty data")
        scale = float(np.std(np.vstack(xy)))
        if scale == 0.0:
            raise ValueError("cannot fit NormStats: all offsets are zero")
        return cls(scale=scale)


def normalize_offsets(offsets: Offsets, stats: NormStats) -> Offsets:
    """Divide ``(dx, dy)`` by ``stats.scale``. Scale-only, so zero stays zero."""
    out = as_array(offsets)
    out[:, :2] /= stats.scale
    return out


def denormalize_offsets(offsets: Offsets, stats: NormStats) -> Offsets:
    """Inverse of :func:`normalize_offsets`."""
    out = as_array(offsets)
    out[:, :2] *= stats.scale
    return out


# --------------------------------------------------------------------------- #
# Geometry on absolute points
# --------------------------------------------------------------------------- #


def bounding_box(points: Points) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)``."""
    points = as_array(points)
    if len(points) == 0:
        raise ValueError("bounding box of empty points is undefined")
    (x0, y0), (x1, y1) = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def translate(points: Points, dx: float, dy: float) -> Points:
    """Shift every point by ``(dx, dy)``."""
    out = as_array(points)
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def scale(points: Points, factor: float, about: tuple[float, float] = (0.0, 0.0)) -> Points:
    """Uniformly scale points by ``factor`` about the point ``about``."""
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    out = as_array(points)
    c = np.asarray(about, dtype=np.float64)
    out[:, :2] = (out[:, :2] - c) * factor + c
    return out


def flip_y(points: Points) -> Points:
    """Mirror vertically (``y -> -y``). For converting from +y-up sources."""
    out = as_array(points)
    out[:, 1] *= -1.0
    return out


def fit_to_box(points: Points, x: float, y: float, width: float, height: float) -> Points:
    """Uniformly scale + translate points so they fit inside the box, top-left aligned."""
    if width <= 0 or height <= 0:
        raise ValueError("box width and height must be positive")
    x0, y0, x1, y1 = bounding_box(points)
    w, h = x1 - x0, y1 - y0
    factors = [f for f in (width / w if w > 0 else None, height / h if h > 0 else None) if f]
    factor = min(factors) if factors else 1.0
    out = translate(points, -x0, -y0)
    out = scale(out, factor)
    return translate(out, x, y)
