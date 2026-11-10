import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hwrobot import strokes as S  # noqa: E402
from hwrobot.model import hs_bridge  # noqa: E402
from hwrobot.model.base import Style  # noqa: E402
from hwrobot.model.network import HandwritingRNN, TFLSTMCell  # noqa: E402
from hwrobot.model.synth import (  # noqa: E402
    _spread_words, _trim_past_end, neatness_to_bias, split_chunks, xheight_band,
)

needs_weights = pytest.mark.skipif(not hs_bridge.available(), reason="pretrained weights not converted")


# ---------------------------------------------------------------- network (tiny, random)

def tiny():
    torch.manual_seed(0)
    return HandwritingRNN(lstm_size=8, attn_components=2, out_components=3)


def test_tf_lstm_matches_formula():
    cell = TFLSTMCell(3, 2)
    x, h, c = torch.randn(1, 3), torch.randn(1, 2), torch.randn(1, 2)
    i, j, f, o = (torch.cat([x, h], 1) @ cell.kernel + cell.bias).chunk(4, 1)
    c_ref = torch.sigmoid(f + 1) * c + torch.sigmoid(i) * torch.tanh(j)
    h2, c2 = cell(x, h, c)
    torch.testing.assert_close(c2, c_ref)
    torch.testing.assert_close(h2, torch.sigmoid(o) * torch.tanh(c_ref))


def test_nll_is_finite_and_trains():
    net = tiny()
    B, T, U = 2, 12, 5
    x = torch.randn(B, T, 3)
    x[..., 2] = (torch.rand(B, T) > 0.8).float()
    y = torch.roll(x, -1, 1)
    chars = torch.nn.functional.one_hot(torch.randint(1, 73, (B, U)), 73).float()
    loss = net.nll(x, y, torch.tensor([T, T - 3]), chars, torch.ones(B, U))
    assert torch.isfinite(loss)
    loss.backward()
    assert net.lstm1.kernel.grad is not None


def test_sample_shapes_and_priming():
    net = tiny()
    chars = torch.tensor([[24, 47, 0, 0], [24, 47, 48, 0]])
    lens = torch.tensor([3, 4])
    out = net.sample(chars, lens, torch.zeros(2), max_steps=30, generator=torch.Generator().manual_seed(1))
    assert len(out) == 2
    for offs, idx in out:
        assert offs.ndim == 2 and offs.shape[1] == 3 and len(offs) <= 30 and len(idx) == len(offs)
    primed = net.sample(chars, lens, torch.zeros(2), torch.randn(2, 6, 3), torch.tensor([6, 4]), max_steps=10)
    assert len(primed) == 2


def test_npz_roundtrip(tmp_path):
    net = tiny()
    path = tmp_path / "w.npz"
    np.savez(path, **{k: v.detach().numpy() for k, v in net.state_dict().items()})
    net2 = HandwritingRNN.from_npz(path)
    for k, v in net.state_dict().items():
        torch.testing.assert_close(net2.state_dict()[k], v)


# ---------------------------------------------------------------- synth helpers

def test_split_chunks():
    assert split_chunks("a b c", 3) == ["a b", "c"]
    assert split_chunks("", 10) == []
    assert all(len(c) <= 5 for c in split_chunks("abcdefghijkl xy", 5))
    assert " ".join(split_chunks("the quick brown fox", 9)) == "the quick brown fox"


def test_xheight_band_finds_dense_zone():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.uniform(-1, 0, 800), rng.uniform(-2.5, 1.2, 100)])
    top, base = xheight_band(y)
    assert -1.2 < top < -0.6 and -0.4 < base < 0.2


def test_trim_past_end_drops_trailing_scribble():
    offs = S.from_strokes([[(0, 0), (1, 0)], [(2, 0), (3, 0)], [(4, 0)]])
    idx = np.array([0, 1, 2, 3, 9])
    o, i = _trim_past_end(offs, idx, n_chars=5)
    assert len(o) == 4 and len(i) == 4


def test_spread_words_shifts_later_words():
    pts = S.from_strokes([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
    out = _spread_words(pts, np.array([0, 0, 2, 2]), "a b", extra=5.0)
    assert out[0, 0] == 0 and out[2, 0] == 7


def test_track_letters_spreads_letters_and_keeps_dots_over_their_letter():
    from hwrobot.model.synth import _track_letters

    body = np.column_stack([np.linspace(0, 10, 30), np.zeros(30), np.r_[np.zeros(29), 1]])
    dot = np.array([[2.0, -3.0, 1.0]])
    pts = np.vstack([body, dot])
    idx = np.r_[np.repeat(np.arange(5), 6), 4]  # dot drawn when attention is already at letter 4
    out = _track_letters(pts, idx, extra=1.0)
    assert out[29, 0] > pts[29, 0] + 3  # end of word pushed right by ~4 letters
    assert abs(out[30, 0] - out[np.argmin(np.abs(pts[:30, 0] - 2.0)), 0]) < 0.6  # dot follows its letter
    assert np.all(np.diff(out[:30, 0]) > 0)  # no backwards jumps


def test_baseline_fit_levels_a_tilted_line():
    from hwrobot.model.synth import straighten

    # 'n'-like arches sitting on y=0 (+y down): equal time spent low and high, like real letters
    t = np.linspace(0, 40, 800)
    y = -(1 - np.cos(t * 2)) / 2
    tilted = np.column_stack([t, y + 0.05 * t, np.zeros_like(t)])
    tilted[-1, 2] = 1
    flat, h = straighten(tilted)
    bottoms = flat[np.cos(t * 2) > 0.999]
    assert np.ptp(bottoms[:, 1]) < 0.15 and abs(np.median(bottoms[:, 1])) < 0.1


def test_bias_mapping():
    assert neatness_to_bias(0) == 0 and neatness_to_bias(1) == pytest.approx(1.5)


# ---------------------------------------------------------------- the real pretrained model

@pytest.fixture(scope="module")
def synth():
    from hwrobot.model.synth import ModelSynth
    return ModelSynth()


@needs_weights
def test_real_model_writes_at_requested_size(synth):
    style = Style(x_height_mm=3.0, seed=3)
    pts = synth.write("handwriting on real paper", style)
    S.as_array(pts)
    from hwrobot.model.synth import baseline_points

    top, base = xheight_band(pts[:, 1])
    assert 2.0 < base - top < 4.2
    bottoms = baseline_points(pts, base - top)
    assert abs(np.median(bottoms[:, 1])) < 0.3  # letters sit on y=0 (the rule)
    x0, _, x1, _ = S.bounding_box(pts)
    assert 25 < x1 - x0 < 120


@needs_weights
def test_real_model_is_cached_and_batches(synth):
    style = Style(seed=5)
    a, b = synth.write_many(["first line here", "second line"], style)
    again = synth.write("first line here", style)
    np.testing.assert_array_equal(a, again)
    assert len(b)


@needs_weights
def test_real_model_reports_unsupported(synth):
    synth.write("Quiz & Zoo €", Style())
    assert {"Q", "&", "Z", "€"} <= synth.last_unsupported


@needs_weights
def test_styles_listed(synth):
    ids = [h["id"] for h in synth.hands()]
    assert "hs-0" in ids and "hs-12" in ids
