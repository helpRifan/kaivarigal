"""Simulator plotter — the default backend. Renders strokes to PNG / SVG.

Behaves like the real plotter (same input, same bounds check) but draws onto
an in-memory page instead of moving motors. Optionally shows pen-up travel
moves as faint dashed lines, which is handy for spotting wasteful paths.

Run as a script for a demo::

    python -m hwrobot.plotter out/demo.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hwrobot import strokes as S
from hwrobot.plotter.base import A4_MM, Plotter

INK = (20, 40, 120)        # blue ballpoint-ish
TRAVEL = (230, 120, 120)   # pen-up moves
PAPER = (255, 255, 255)


class SimulatorPlotter(Plotter):
    """Accumulates strokes and renders them.

    Args:
        page_size_mm: page width, height in mm.
        px_per_mm: render resolution (4 px/mm ≈ 100 dpi, 12 ≈ 300 dpi).
        pen_width_mm: drawn line width.
        show_travel: draw pen-up moves between strokes.
    """

    def __init__(
        self,
        page_size_mm: tuple[float, float] = A4_MM,
        px_per_mm: float = 4.0,
        pen_width_mm: float = 0.5,
        show_travel: bool = False,
    ) -> None:
        super().__init__(page_size_mm)
        if px_per_mm <= 0 or pen_width_mm <= 0:
            raise ValueError("px_per_mm and pen_width_mm must be positive")
        self.px_per_mm = float(px_per_mm)
        self.pen_width_mm = float(pen_width_mm)
        self.show_travel = show_travel
        self._strokes: list[np.ndarray] = []  # each (M, 2) in mm

    # -- Plotter ------------------------------------------------------------ #

    def _plot(self, points: S.Points) -> None:
        self._strokes.extend(S.split_strokes(points))

    # -- inspection --------------------------------------------------------- #

    @property
    def strokes(self) -> list[np.ndarray]:
        """Copies of the pen-down strokes drawn so far, each ``(M, 2)`` mm."""
        return [s.copy() for s in self._strokes]

    def clear(self) -> None:
        self._strokes.clear()

    def travel_moves(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pen-up moves: (from, to) pairs between consecutive strokes, starting at (0, 0)."""
        moves = []
        pos = np.zeros(2)
        for s in self._strokes:
            moves.append((pos, s[0]))
            pos = s[-1]
        return moves

    def ink_length_mm(self) -> float:
        """Total pen-down distance — useful for time / ink estimates."""
        return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in self._strokes))

    # -- rendering ---------------------------------------------------------- #

    def render(self) -> Image.Image:
        """Return the page as an RGB Pillow image."""
        k = self.px_per_mm
        w, h = self.page_size_mm
        img = Image.new("RGB", (max(1, round(w * k)), max(1, round(h * k))), PAPER)
        draw = ImageDraw.Draw(img)
        width_px = max(1, round(self.pen_width_mm * k))
        r = width_px / 2

        if self.show_travel:
            for a, b in self.travel_moves():
                _dashed_line(draw, a * k, b * k, TRAVEL, dash_px=max(2, 2 * k))

        for s in self._strokes:
            px = [tuple(p) for p in (s * k)]
            if len(px) == 1 or np.allclose(s, s[0]):
                x, y = px[0]
                draw.ellipse([x - r, y - r, x + r, y + r], fill=INK)
            else:
                draw.line(px, fill=INK, width=width_px, joint="curve")
                for x, y in (px[0], px[-1]):  # round caps
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=INK)
        return img

    def save_png(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render().save(path, format="PNG", dpi=(self.px_per_mm * 25.4,) * 2)
        return path

    def to_svg(self) -> str:
        """SVG in mm units (1 user unit = 1 mm), so it prints at true size."""
        w, h = self.page_size_mm
        ink = "#%02x%02x%02x" % INK
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" '
            f'viewBox="0 0 {w:g} {h:g}">',
            f'<rect width="{w:g}" height="{h:g}" fill="white"/>',
            f'<g fill="none" stroke="{ink}" stroke-width="{self.pen_width_mm:g}" '
            f'stroke-linecap="round" stroke-linejoin="round">',
        ]
        for s in self._strokes:
            if len(s) == 1:
                s = np.vstack([s, s])  # zero-length path + round cap = dot
            d = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in s)
            parts.append(f'<path d="{d}"/>')
        parts.append("</g></svg>")
        return "\n".join(parts)

    def save_svg(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg(), encoding="utf-8")
        return path


def _dashed_line(draw: ImageDraw.ImageDraw, a: np.ndarray, b: np.ndarray, fill, dash_px: float) -> None:
    length = float(np.linalg.norm(b - a))
    if length == 0:
        return
    n = max(1, int(length // dash_px))
    for i in range(0, n, 2):
        p = a + (b - a) * (i / n)
        q = a + (b - a) * (min(i + 1, n) / n)
        draw.line([tuple(p), tuple(q)], fill=fill, width=1)


def demo_points() -> S.Points:
    """A few synthetic strokes (a wavy 'signature', a loop, a dot) for demos/tests."""
    t = np.linspace(0, 4 * np.pi, 200)
    wave = np.column_stack([30 + t * 8, 40 + 5 * np.sin(t) + 2 * np.sin(3 * t)])
    a = np.linspace(0, 2 * np.pi, 80)
    loop = np.column_stack([150 + 10 * np.cos(a), 40 + 6 * np.sin(a)])
    dot = [(170, 30)]
    return S.from_strokes([wave, loop, dot])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path("out/demo.png")
    sim = SimulatorPlotter(show_travel=True)
    sim.plot(demo_points())
    png = sim.save_png(out)
    svg = sim.save_svg(out.with_suffix(".svg"))
    print(f"wrote {png} and {svg} ({sim.ink_length_mm():.0f} mm of ink)")
    return 0
