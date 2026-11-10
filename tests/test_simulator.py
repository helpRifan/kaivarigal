import numpy as np
import pytest
from PIL import Image

from hwrobot import strokes as S
from hwrobot.plotter import OutOfBoundsError, Plotter, SimulatorPlotter
from hwrobot.plotter.simulator import INK, PAPER, demo_points, main

SQUARE = S.from_strokes([[(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]])


def test_is_a_plotter():
    assert isinstance(SimulatorPlotter(), Plotter)


def test_render_size_matches_page_and_resolution():
    img = SimulatorPlotter(page_size_mm=(100, 50), px_per_mm=2).render()
    assert img.size == (200, 100)


def test_blank_page_is_white():
    arr = np.asarray(SimulatorPlotter(page_size_mm=(20, 20)).render())
    assert (arr == PAPER).all()


def test_ink_lands_where_stroke_is():
    sim = SimulatorPlotter(page_size_mm=(50, 50), px_per_mm=4)
    sim.plot(SQUARE)
    arr = np.asarray(sim.render())
    # On the top edge of the square (20 mm, 10 mm) -> pixel (80, 40): inked.
    assert tuple(arr[40, 80]) == INK
    # Centre of the square (20 mm, 20 mm): untouched paper.
    assert tuple(arr[80, 80]) == PAPER
    # Outside the square: paper.
    assert tuple(arr[180, 180]) == PAPER


def test_single_point_stroke_draws_a_dot():
    sim = SimulatorPlotter(page_size_mm=(20, 20), px_per_mm=10, pen_width_mm=1)
    sim.plot([[10, 10, 1]])
    arr = np.asarray(sim.render())
    assert tuple(arr[100, 100]) == INK


def test_out_of_bounds_raises_and_draws_nothing():
    sim = SimulatorPlotter(page_size_mm=(50, 50))
    with pytest.raises(OutOfBoundsError):
        sim.plot([[10, 10, 0], [60, 10, 1]])
    with pytest.raises(OutOfBoundsError):
        sim.plot([[-1, 10, 1]])
    assert sim.strokes == []


def test_invalid_points_rejected():
    with pytest.raises(ValueError):
        SimulatorPlotter().plot([[10, 10, 0]])  # stroke never ends


def test_plot_accumulates_and_clear():
    sim = SimulatorPlotter()
    sim.plot(SQUARE)
    sim.plot([[100, 100, 1]])
    assert len(sim.strokes) == 2
    sim.clear()
    assert sim.strokes == []


def test_ink_length_and_travel():
    sim = SimulatorPlotter()
    sim.plot(SQUARE)
    assert sim.ink_length_mm() == pytest.approx(80.0)
    (start, end), = sim.travel_moves()
    np.testing.assert_array_equal(start, [0, 0])
    np.testing.assert_array_equal(end, [10, 10])


def test_save_png_roundtrip(tmp_path):
    sim = SimulatorPlotter(page_size_mm=(50, 50), px_per_mm=4, show_travel=True)
    sim.plot(SQUARE)
    path = sim.save_png(tmp_path / "sub" / "page.png")
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size == (200, 200)


def test_svg_is_true_size(tmp_path):
    sim = SimulatorPlotter()
    sim.plot(SQUARE)
    sim.plot([[50, 50, 1]])
    svg = sim.save_svg(tmp_path / "page.svg").read_text(encoding="utf-8")
    assert 'width="210mm"' in svg and 'height="297mm"' in svg
    assert svg.count("<path") == 2


def test_context_manager():
    with SimulatorPlotter() as sim:
        sim.plot(SQUARE)
    assert len(sim.strokes) == 1


def test_demo_fits_on_a4_and_main_writes_files(tmp_path, capsys):
    SimulatorPlotter().check_bounds(demo_points())
    assert main([str(tmp_path / "demo.png")]) == 0
    assert (tmp_path / "demo.png").exists()
    assert (tmp_path / "demo.svg").exists()
