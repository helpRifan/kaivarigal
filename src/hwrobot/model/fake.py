"""Placeholder handwriting: joined-up scribble loops, NOT my handwriting.

Lets the whole app run before the real model exists. It respects every :class:`Style` control
(size, slant, neatness, spacing) and the letter classes (ascenders, descenders, capitals, i/j
dots, t crosses), so layout and preview behave realistically. Deterministic per (text, seed).
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from hwrobot import strokes as S
from hwrobot.model.base import Style, word_gap_mm

ASCENDERS = set("bdfhklt")
DESCENDERS = set("gjpqy")
WIDTH = {"m": 1.35, "w": 1.3, "i": 0.42, "l": 0.45, "j": 0.5, "t": 0.55, "f": 0.6, "r": 0.6, "s": 0.65}
SAMPLES = 10
BASE_SLANT_DEG = 8.0  # this fake hand's own slant; Style.slant_deg adds to it


class FakeSynth:
    name = "Placeholder scribble (model not trained yet)"
    simulated = True

    def write(self, text: str, style: Style) -> S.Points:
        return self._render(text, style)[0]

    def measure(self, text: str, style: Style) -> float:
        return self._render(text, style)[1]

    # ------------------------------------------------------------------ #

    def _render(self, text: str, style: Style) -> tuple[S.Points, float]:
        xh = style.x_height_mm
        mess = 1.0 - min(max(style.neatness, 0.0), 1.0)
        rng = np.random.default_rng([style.seed & 0xFFFFFFFF, zlib.crc32(text.encode())])
        t = np.linspace(0.0, 1.0, SAMPLES)

        strokes: list[np.ndarray] = []
        extras: list[np.ndarray] = []
        current: list[np.ndarray] = []
        x = 0.0

        def bump(x0: float, w: float, h: float, loop: float) -> np.ndarray:
            px = x0 + w * t - loop * w * 0.3 * np.sin(2 * np.pi * t)
            py = -h * np.sin(np.pi * t) ** 2
            return np.column_stack([px, py])

        def end_stroke() -> None:
            if current:
                strokes.append(np.vstack(current))
                current.clear()

        for ch in text:
            if ch.isspace():
                end_stroke()
                x += word_gap_mm(style)
                continue

            hvar = 1.0 + rng.normal(0.0, 0.12 * mess + 0.02)  # each letter's height differs
            w = xh * WIDTH.get(ch.lower(), 0.8) * style.letter_spacing

            if ch in ".,:;'\"!?-":
                end_stroke()
                w = xh * 0.45
                cx = x + w * 0.4
                if ch in ".:;!?":
                    extras.append(np.array([[cx, 0.0]]))
                if ch in ":;":
                    extras.append(np.array([[cx, -0.9 * xh]]))
                if ch in ",;":
                    extras.append(np.array([[cx, 0.0], [cx - 0.15 * xh, 0.45 * xh]]))
                if ch in "'\"":
                    extras.append(np.array([[cx, -2.0 * xh], [cx - 0.1 * xh, -1.5 * xh]]))
                if ch in "!?":
                    extras.append(np.array([[cx, -2.0 * xh], [cx, -0.5 * xh]]))
                if ch == "-":
                    extras.append(np.array([[x, -0.5 * xh], [x + w, -0.5 * xh]]))
                x += w + 0.1 * xh
                continue

            if ch.isupper() or ch.isdigit():
                pts = bump(x, w * 1.1, 2.0 * xh * hvar, 0.6 if ch.isupper() else 0.2)
            elif ch in ASCENDERS:
                pts = bump(x, w, 2.1 * xh * hvar, 0.5)
            elif ch in DESCENDERS:
                body = bump(x, w * 0.6, xh * hvar, 0.3)
                tail = bump(x + w * 0.6, w * 0.4, -1.1 * xh * hvar, 0.6)
                pts = np.vstack([body, tail[1:]])
            else:
                pts = bump(x, w, xh * hvar, 0.35)

            pts = pts + rng.normal(0.0, 0.04 * xh * mess + 0.004 * xh, pts.shape)
            current.append(pts if not current else pts[1:])

            if ch in "ij":
                extras.append(np.array([[x + w * 0.45, -1.7 * xh]]))
            if ch == "t":
                extras.append(np.array([[x - 0.1 * w, -1.3 * xh], [x + w, -1.35 * xh]]))

            x += w + 0.08 * xh * style.letter_spacing

        end_stroke()
        all_strokes = strokes + extras
        if not all_strokes:
            return np.zeros((0, 3)), 0.0

        pts = S.from_strokes(all_strokes)
        pts[:, 0] -= pts[:, 1] * math.tan(math.radians(BASE_SLANT_DEG + style.slant_deg))
        return pts, x
