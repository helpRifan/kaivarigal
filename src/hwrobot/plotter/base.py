"""Plotter interface.

Every backend (simulator now, GRBL later) takes the same input: absolute-mm
points from :mod:`hwrobot.strokes` (origin top-left of the page, +y down).
Nothing here touches hardware, so this always imports on a plain laptop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hwrobot import strokes as S

#: A4 portrait, millimetres.
A4_MM = (210.0, 297.0)


class OutOfBoundsError(ValueError):
    """Raised when strokes would leave the page — never send these to a real pen."""


class Plotter(ABC):
    """Base class for plotter backends.

    Subclasses implement :meth:`_plot`. Callers use :meth:`plot`, which
    validates the points and checks they fit on the page first.
    """

    def __init__(self, page_size_mm: tuple[float, float] = A4_MM) -> None:
        w, h = page_size_mm
        if w <= 0 or h <= 0:
            raise ValueError("page size must be positive")
        self.page_size_mm = (float(w), float(h))

    def check_bounds(self, points: S.Points) -> None:
        """Raise :class:`OutOfBoundsError` if any point is off the page."""
        if len(points) == 0:
            return
        x0, y0, x1, y1 = S.bounding_box(points)
        w, h = self.page_size_mm
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            raise OutOfBoundsError(
                f"strokes span x[{x0:.1f}, {x1:.1f}] y[{y0:.1f}, {y1:.1f}] mm, "
                f"page is {w:.0f} x {h:.0f} mm"
            )

    def plot(self, points: S.Points) -> None:
        """Draw absolute-mm points. Can be called repeatedly to add more strokes."""
        pts = S.as_array(points)
        self.check_bounds(pts)
        if len(pts):
            self._plot(pts)

    @abstractmethod
    def _plot(self, points: S.Points) -> None:
        """Backend-specific drawing. ``points`` is already validated and in bounds."""

    def close(self) -> None:
        """Release resources (serial port, etc.). Default: nothing to do."""

    def __enter__(self) -> "Plotter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
