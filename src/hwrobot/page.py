"""A page to write on: its size and its ruled lines, in absolute mm (origin top-left, +y down).

Produced by :mod:`hwrobot.vision` (from a photo or a template), consumed by
:mod:`hwrobot.layout`. The ruled line's ``y_mm`` is where the text's baseline sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class RuledLine:
    id: str
    y_mm: float
    x_start_mm: float
    x_end_mm: float
    has_writing: bool = False

    @property
    def width_mm(self) -> float:
        return self.x_end_mm - self.x_start_mm

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "y_mm": self.y_mm,
            "x_start_mm": self.x_start_mm,
            "x_end_mm": self.x_end_mm,
            "has_writing": self.has_writing,
        }


@dataclass
class Page:
    id: str
    label: str
    width_mm: float
    height_mm: float
    lines: list[RuledLine]
    margin_x_mm: float | None = None
    #: True when the lines were guessed rather than detected (fake vision).
    simulated: bool = False
    #: Flattened, top-down image of the page. Not serialized.
    image: Image.Image | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.lines.sort(key=lambda l: l.y_mm)

    def line(self, line_id: str) -> RuledLine:
        for l in self.lines:
            if l.id == line_id:
                return l
        raise KeyError(line_id)

    @property
    def line_spacing_mm(self) -> float | None:
        if len(self.lines) < 2:
            return None
        ys = [l.y_mm for l in self.lines]
        gaps = sorted(b - a for a, b in zip(ys, ys[1:]))
        return gaps[len(gaps) // 2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "margin_x_mm": self.margin_x_mm,
            "line_spacing_mm": self.line_spacing_mm,
            "simulated": self.simulated,
            "lines": [l.to_dict() for l in self.lines],
        }
