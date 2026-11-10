import numpy as np
import pytest

from hwrobot import strokes as S
from hwrobot.model.base import PRESETS, Style
from hwrobot.model.fake import FakeSynth

synth = FakeSynth()


def test_output_is_valid_points():
    pts = synth.write("Hello, world! Testing jiggly t's.", Style())
    S.as_array(pts)  # raises if invalid
    assert len(pts) > 50


def test_empty_and_spaces():
    assert synth.write("", Style()).shape == (0, 3)
    assert synth.write("   ", Style()).shape == (0, 3)
    assert synth.measure("", Style()) == 0


def test_deterministic_per_seed():
    a = synth.write("same text", Style(seed=1))
    b = synth.write("same text", Style(seed=1))
    c = synth.write("same text", Style(seed=2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_width_grows_with_text_and_spacing():
    s = Style()
    assert synth.measure("hello world", s) > synth.measure("hello", s)
    assert synth.measure("hello", s.with_(letter_spacing=1.4)) > synth.measure("hello", s)


def test_size_scales_height():
    small = S.bounding_box(synth.write("hello", Style(x_height_mm=2.0, neatness=1)))
    big = S.bounding_box(synth.write("hello", Style(x_height_mm=4.0, neatness=1)))
    assert (big[3] - big[1]) > 1.5 * (small[3] - small[1])


def test_descenders_go_below_baseline_ascenders_above():
    _, y0, _, y1 = S.bounding_box(synth.write("gy", Style(neatness=1)))
    assert y1 > 1.0  # below baseline (+y down)
    _, y0, _, _ = S.bounding_box(synth.write("lk", Style(neatness=1)))
    assert y0 < -4.0  # tall


def test_i_dot_is_separate_stroke():
    assert len(S.split_strokes(synth.write("i", Style()))) == 2


@pytest.mark.parametrize("key", list(PRESETS))
def test_presets_render(key):
    _, style = PRESETS[key]
    assert len(synth.write("Preset check", style))
