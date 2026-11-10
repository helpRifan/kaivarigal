"""The handwriting-synthesis interface every generator implements (fake now, real model later).

A synth turns one line of text into absolute-mm points (see :mod:`hwrobot.strokes`) with the
origin at the **start of the baseline**: x grows to the right, ascenders have negative y,
descenders positive y. Layout moves the result onto a ruled line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Protocol

from hwrobot import strokes as S


@dataclass(frozen=True)
class Style:
    """Controls exposed in the UI. The real model maps ``neatness`` to its sampling bias."""

    x_height_mm: float = 2.6      # height of a lowercase 'x'
    slant_deg: float = 8.0        # positive leans right
    neatness: float = 0.7         # 0 = rushed and varied, 1 = careful and even
    letter_spacing: float = 1.0   # multiplier
    word_spacing: float = 1.0     # multiplier
    seed: int = 0                 # same seed + same text = same result

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "Style":
        return replace(self, **changes)


#: Style profiles shown in the UI.
PRESETS: dict[str, tuple[str, Style]] = {
    "normal": ("Normal", Style()),
    "exam": ("Exam (fast)", Style(x_height_mm=2.4, slant_deg=13, neatness=0.4,
                                  letter_spacing=0.92, word_spacing=0.9)),
    "neat": ("Neat notes", Style(x_height_mm=2.8, slant_deg=5, neatness=0.95,
                                 letter_spacing=1.05, word_spacing=1.1)),
}


def word_gap_mm(style: Style) -> float:
    """Space between words."""
    return 1.1 * style.x_height_mm * style.word_spacing


class HandwritingSynth(Protocol):
    #: Shown in the UI so it's always clear what is generating the writing.
    name: str
    #: True for placeholders that don't produce my real handwriting.
    simulated: bool

    def write(self, text: str, style: Style) -> S.Points:
        """Render one line of text. Empty text returns an empty ``(0, 3)`` array."""
        ...

    def measure(self, text: str, style: Style) -> float:
        """Advance width in mm of ``text`` (used for word wrapping)."""
        ...
