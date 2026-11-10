import pytest

from hwrobot import strokes as S
from hwrobot.layout.engine import layout_page
from hwrobot.model.base import Style
from hwrobot.model.fake import FakeSynth
from hwrobot.vision.fake import make_page

synth = FakeSynth()
STYLE = Style(neatness=1)
LONG = " ".join(["handwriting"] * 40)


def ids(result):
    return [l.line_id for l in result.lines]


def test_short_text_stays_on_its_line():
    page = make_page("ruled-8")
    r = layout_page(page, {"L3": "Hello there"}, synth, STYLE)
    assert ids(r) == ["L3"]
    assert r.overflow == "" and r.warnings == []


def test_long_text_wraps_onto_following_lines():
    page = make_page("ruled-8")
    r = layout_page(page, {"L2": LONG}, synth, STYLE)
    assert ids(r)[:3] == ["L2", "L3", "L4"]
    assert r.lines[1].continued
    assert " ".join(l.text for l in r.lines).split() == LONG.split()


def test_text_stays_inside_line_and_page():
    page = make_page("ruled-8")
    r = layout_page(page, {"L1": LONG}, synth, STYLE)
    for placed in r.lines:
        line = page.line(placed.line_id)
        x0, y0, x1, y1 = S.bounding_box(placed.points)
        assert line.x_start_mm <= x0 and x1 <= line.x_end_mm + 0.01
        assert y0 < line.y_mm < y1 + 0.5  # straddles the rule: body above, maybe descenders below


def test_wrap_stops_at_line_with_own_text():
    page = make_page("ruled-8")
    r = layout_page(page, {"L1": LONG, "L3": "mine"}, synth, STYLE)
    assert ids(r) == ["L1", "L2", "L3"]
    assert r.overflow
    assert any("own text" in w for w in r.warnings)


def test_page_full_overflow():
    page = make_page("ruled-8")
    last = page.lines[-1].id
    r = layout_page(page, {last: LONG}, synth, STYLE)
    assert ids(r) == [last]
    assert r.overflow and any("Page is full" in w for w in r.warnings)


def test_skip_used_lines():
    page = make_page("half-filled")
    first_free = next(l for l in page.lines if not l.has_writing)
    r = layout_page(page, {"L1": LONG}, synth, STYLE, skip_used=True)
    assert ids(r)[1] == first_free.id
    r2 = layout_page(page, {"L1": LONG}, synth, STYLE, skip_used=False)
    assert ids(r2)[1] == "L2"


def test_big_letters_warn():
    page = make_page("ruled-7")
    r = layout_page(page, {"L1": "hi"}, synth, Style(x_height_mm=4))
    assert any("large" in w for w in r.warnings)


def test_unknown_line_rejected():
    with pytest.raises(KeyError):
        layout_page(make_page("ruled-8"), {"nope": "x"}, synth, STYLE)


def test_all_points_valid():
    r = layout_page(make_page("ruled-8"), {"L1": LONG}, synth, STYLE)
    S.as_array(r.all_points())
