"""Fake page vision: page templates, and a pretend "scan" of an uploaded photo.

Templates are real and stay useful later (no photo needed for a known page layout).
``scan_photo`` is the fake part: it only resizes the photo to A4 and guesses evenly spaced
lines. Step 3 replaces it with real corner + line detection.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageOps

from hwrobot.model.base import Style
from hwrobot.model.fake import FakeSynth
from hwrobot.page import Page, RuledLine
from hwrobot.strokes import split_strokes

A4 = (210.0, 297.0)
PX_PER_MM = 4.0
PAPER = (250, 248, 242)
RULE = (164, 192, 226)
MARGIN = (221, 118, 112)
OLD_INK = (52, 62, 110)


@dataclass(frozen=True)
class Template:
    id: str
    name: str
    description: str
    spacing: float
    top: float
    margin: float | None
    bottom: float = 14.0
    header: bool = False
    filled_lines: int = 0


TEMPLATES: dict[str, Template] = {
    t.id: t
    for t in [
        Template("ruled-8", "Ruled notebook", "A4, 8 mm lines, red margin", 8.0, 30.0, 25.0),
        Template("ruled-7", "Narrow ruled", "A4, 7 mm lines, red margin", 7.0, 28.0, 25.0),
        Template("lab-record", "Lab record", "Header box, then 8 mm lines", 8.0, 72.0, 20.0, header=True),
        Template("half-filled", "Half-written page", "8 mm lines, first 7 already written", 8.0, 30.0, 25.0,
                 filled_lines=7),
    ]
}

_FILLER = [
    "Aim: to study the properties of light",
    "and verify the laws of reflection using",
    "a plane mirror and pins. Apparatus used",
    "drawing board, white sheet, mirror strip",
    "pins and a protractor. Procedure: fix",
    "the sheet on the board and draw a line",
    "where the mirror will be placed and then",
]


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _lines(width: float, height: float, spacing: float, top: float, margin: float | None,
           bottom: float) -> list[RuledLine]:
    x_start = (margin + 2.0) if margin else 12.0
    out, y, i = [], top, 0
    while y <= height - bottom:
        out.append(RuledLine(f"L{i + 1}", round(y, 2), x_start, width - 10.0))
        y += spacing
        i += 1
    return out


def make_page(template_id: str) -> Page:
    """Build a page (with its rendered image) from a template."""
    tpl = TEMPLATES[template_id]
    w, h = A4
    lines = _lines(w, h, tpl.spacing, tpl.top, tpl.margin, tpl.bottom)
    for line in lines[: tpl.filled_lines]:
        line.has_writing = True
    page = Page(_new_id(), tpl.name, w, h, lines, margin_x_mm=tpl.margin)
    page.image = render_page_image(page, tpl)
    return page


def render_page_image(page: Page, tpl: Template | None = None) -> Image.Image:
    k = PX_PER_MM
    img = Image.new("RGB", (round(page.width_mm * k), round(page.height_mm * k)), PAPER)
    d = ImageDraw.Draw(img)
    for line in page.lines:
        d.line([(0, line.y_mm * k), (img.width, line.y_mm * k)], fill=RULE, width=2)
    if page.margin_x_mm:
        x = page.margin_x_mm * k
        d.line([(x, 0), (x, img.height)], fill=MARGIN, width=2)
    if tpl and tpl.header:
        box = [14 * k, 14 * k, (page.width_mm - 14) * k, 60 * k]
        d.rectangle(box, outline=(90, 90, 90), width=2)
        for i, label in enumerate(["Experiment No.", "Date", "Title"]):
            y = (22 + i * 13) * k
            d.text((18 * k, y - 10), label, fill=(90, 90, 90))
            d.line([(48 * k, y + 4), ((page.width_mm - 18) * k, y + 4)], fill=(170, 170, 170), width=1)
    synth, style = FakeSynth(), Style(seed=7, neatness=0.5)
    for line, text in zip([l for l in page.lines if l.has_writing], _FILLER * 10):
        pts = synth.write(text, style)
        pts[:, 0] += line.x_start_mm + 1
        pts[:, 1] += line.y_mm - 0.4
        for s in split_strokes(pts):
            xy = [tuple(p) for p in s * k]
            if len(xy) > 1:
                d.line(xy, fill=OLD_INK, width=2, joint="curve")
            else:
                x, y = xy[0]
                d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=OLD_INK)
    return img


def scan_photo(data: bytes) -> Page:
    """FAKE: treat the photo as an already-flat A4 page and guess 8 mm lines."""
    try:
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types
        raise ValueError("That file isn't an image Pillow can read.") from exc
    w, h = A4
    img = photo.resize((round(w * PX_PER_MM), round(h * PX_PER_MM)))
    page = Page(_new_id(), "Scanned photo", w, h, _lines(w, h, 8.0, 30.0, 25.0, 14.0),
                margin_x_mm=25.0, simulated=True)
    page.image = img
    return page


def blank_page() -> Page:
    """Plain A4 with no lines (calibration grid)."""
    page = Page(_new_id(), "Blank A4", *A4, lines=[])
    page.image = Image.new("RGB", (round(A4[0] * PX_PER_MM), round(A4[1] * PX_PER_MM)), PAPER)
    return page
