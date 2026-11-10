"""Place handwriting onto a page's ruled lines.

Rules:
- Each line can be given its own text. Text that doesn't fit wraps (at word boundaries) onto the
  following lines, as long as those lines have no text of their own and, with ``skip_used``, no
  existing writing.
- Wrapped text that hits a line with its own text, or runs off the end of the page, is reported
  as overflow instead of being dropped silently.
- A rendered line wider than its ruled line is squeezed to fit (and warned about).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hwrobot import strokes as S
from hwrobot.model.base import HandwritingSynth, Style, word_gap_mm
from hwrobot.page import Page, RuledLine

#: Gap between the start of the ruled line and the first letter.
INDENT_MM = 1.0
#: Baseline sits this far above the printed rule.
BASELINE_LIFT_MM = 0.4


@dataclass
class PlacedLine:
    line_id: str
    text: str
    points: S.Points          # absolute page mm
    continued: bool = False   # True when this text wrapped here from an earlier line


@dataclass
class LayoutResult:
    lines: list[PlacedLine] = field(default_factory=list)
    overflow: str = ""
    warnings: list[str] = field(default_factory=list)

    def all_points(self) -> S.Points:
        chunks = [l.points for l in self.lines if len(l.points)]
        return np.vstack(chunks) if chunks else np.zeros((0, 3))


def layout_page(
    page: Page,
    assignments: dict[str, str],
    synth: HandwritingSynth,
    style: Style,
    skip_used: bool = True,
) -> LayoutResult:
    result = LayoutResult()
    unknown = set(assignments) - {l.id for l in page.lines}
    if unknown:
        raise KeyError(f"unknown line ids: {sorted(unknown)}")

    spacing = page.line_spacing_mm
    if spacing and 2.1 * style.x_height_mm > spacing * 0.95:
        result.warnings.append(
            f"Letters ({style.x_height_mm:.1f} mm) are large for {spacing:.1f} mm lines; "
            "tall letters may touch the line above."
        )

    carry: list[str] = []
    for line in page.lines:
        own = assignments.get(line.id, "").split()
        if own:
            if carry:
                result.overflow += (" " if result.overflow else "") + " ".join(carry)
                result.warnings.append(
                    f"Text ran into a line that already has its own text; {len(carry)} word(s) didn't fit."
                )
                carry = []
            words, continued = own, False
        elif carry and not (skip_used and line.has_writing):
            words, continued = carry, True
        else:
            continue

        fitted, carry = _fit_words(words, line, synth, style)
        if fitted:
            text = " ".join(fitted)
            result.lines.append(PlacedLine(line.id, text, _place(text, line, synth, style, result), continued))

    if carry:
        result.overflow += (" " if result.overflow else "") + " ".join(carry)
        result.warnings.append(f"Page is full; {len(carry)} word(s) didn't fit.")
    return result


def _fit_words(
    words: list[str], line: RuledLine, synth: HandwritingSynth, style: Style
) -> tuple[list[str], list[str]]:
    """Greedy wrap: the most leading words that fit on ``line``. Always takes at least one."""
    avail = line.width_mm - INDENT_MM
    gap = word_gap_mm(style)
    used = 0.0
    for i, word in enumerate(words):
        w = synth.measure(word, style)
        need = w if i == 0 else used + gap + w
        if i > 0 and need > avail:
            return words[:i], words[i:]
        used = need
    return words, []


def _place(text: str, line: RuledLine, synth: HandwritingSynth, style: Style, result: LayoutResult) -> S.Points:
    pts = synth.write(text, style)
    if len(pts) == 0:
        return pts
    avail = line.width_mm - INDENT_MM
    x0, _, x1, _ = S.bounding_box(pts)
    width = x1 - min(x0, 0.0)
    if width > avail:
        pts = S.scale(pts, avail / width)
        if width > avail * 1.15:
            result.warnings.append(f"“{text[:24]}” was squeezed to fit its line.")
    x0 = S.bounding_box(pts)[0]
    shift = INDENT_MM - min(x0, 0.0)
    return S.translate(pts, line.x_start_mm + shift, line.y_mm - BASELINE_LIFT_MM)
