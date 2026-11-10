import numpy as np
import pytest

from hwrobot import strokes as S

# Two strokes: an "L" then a dot.
L_AND_DOT = [
    [10.0, 10.0, 0],
    [10.0, 20.0, 0],
    [15.0, 20.0, 1],
    [20.0, 15.0, 1],
]


def test_as_array_accepts_valid_points():
    arr = S.as_array(L_AND_DOT)
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float64


def test_as_array_empty():
    assert S.as_array([]).shape == (0, 3)


@pytest.mark.parametrize(
    "bad, msg",
    [
        ([[1, 2]], "shape"),
        ([[1, 2, 0.5]], "pen_up"),
        ([[1, np.nan, 1]], "NaN"),
        ([[0, 0, 0], [1, 1, 0]], "last point"),
    ],
)
def test_as_array_rejects_invalid(bad, msg):
    with pytest.raises(ValueError, match=msg):
        S.as_array(bad)


def test_as_array_does_not_alias_input():
    src = np.array(L_AND_DOT, dtype=np.float64)
    out = S.as_array(src)
    out[0, 0] = 999
    assert src[0, 0] == 10.0


def test_from_strokes_and_split_roundtrip():
    strokes = [[(10, 10), (10, 20), (15, 20)], [], [(20, 15)]]
    pts = S.from_strokes(strokes)
    np.testing.assert_array_equal(pts, np.array(L_AND_DOT, dtype=float))
    parts = S.split_strokes(pts)
    assert len(parts) == 2
    np.testing.assert_array_equal(parts[0], [[10, 10], [10, 20], [15, 20]])
    np.testing.assert_array_equal(parts[1], [[20, 15]])


def test_split_strokes_empty():
    assert S.split_strokes([]) == []


def test_offsets_roundtrip_is_exact():
    pts = S.as_array(L_AND_DOT)
    offs = S.points_to_offsets(pts)
    # First row relative to origin; then deltas.
    np.testing.assert_array_equal(offs[0], [10, 10, 0])
    np.testing.assert_array_equal(offs[1], [0, 10, 0])
    np.testing.assert_array_equal(offs[3], [5, -5, 1])
    np.testing.assert_allclose(S.offsets_to_points(offs), pts)


def test_offsets_to_points_with_origin():
    offs = S.points_to_offsets(L_AND_DOT)
    moved = S.offsets_to_points(offs, origin=(100, 50))
    np.testing.assert_allclose(moved[:, :2], S.as_array(L_AND_DOT)[:, :2] + [100, 50])


def test_pen_up_preserved_through_conversions():
    pts = S.as_array(L_AND_DOT)
    offs = S.points_to_offsets(pts)
    np.testing.assert_array_equal(offs[:, 2], pts[:, 2])


def test_normalize_roundtrip():
    offs = S.points_to_offsets(L_AND_DOT)
    stats = S.NormStats.fit([offs])
    norm = S.normalize_offsets(offs, stats)
    assert np.std(norm[:, :2]) == pytest.approx(1.0)
    np.testing.assert_array_equal(norm[:, 2], offs[:, 2])  # pen_up untouched
    np.testing.assert_allclose(S.denormalize_offsets(norm, stats), offs)


def test_normstats_rejects_degenerate():
    with pytest.raises(ValueError):
        S.NormStats.fit([])
    with pytest.raises(ValueError):
        S.NormStats.fit([[[0, 0, 1]]])


def test_bounding_box():
    assert S.bounding_box(L_AND_DOT) == (10.0, 10.0, 20.0, 20.0)
    with pytest.raises(ValueError):
        S.bounding_box([])


def test_translate_and_scale():
    pts = S.translate(L_AND_DOT, -10, -10)
    assert S.bounding_box(pts) == (0.0, 0.0, 10.0, 10.0)
    big = S.scale(pts, 2.0)
    assert S.bounding_box(big) == (0.0, 0.0, 20.0, 20.0)
    with pytest.raises(ValueError):
        S.scale(pts, 0)


def test_flip_y():
    flipped = S.flip_y(L_AND_DOT)
    np.testing.assert_array_equal(flipped[:, 1], -S.as_array(L_AND_DOT)[:, 1])


def test_fit_to_box_keeps_aspect():
    # 10x10 shape into a 40x20 box -> limited by height -> 20x20 at (5, 7)
    fitted = S.fit_to_box(L_AND_DOT, x=5, y=7, width=40, height=20)
    x0, y0, x1, y1 = S.bounding_box(fitted)
    assert (x0, y0) == pytest.approx((5, 7))
    assert (x1 - x0, y1 - y0) == pytest.approx((20, 20))


def test_fit_to_box_horizontal_line():
    line = [[0, 5, 0], [10, 5, 1]]  # zero height
    fitted = S.fit_to_box(line, 0, 0, 50, 10)
    assert S.bounding_box(fitted) == pytest.approx((0, 0, 50, 0))


def test_functions_do_not_mutate_input():
    src = np.array(L_AND_DOT, dtype=float)
    before = src.copy()
    S.translate(src, 1, 1)
    S.scale(src, 3)
    S.flip_y(src)
    S.points_to_offsets(src)
    S.normalize_offsets(src, S.NormStats(scale=2.0))
    np.testing.assert_array_equal(src, before)
