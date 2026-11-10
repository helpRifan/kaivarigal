"""Real handwriting: the handwriting-synthesis model behind the :class:`HandwritingSynth` interface.

Pipeline per line of text:
1. Split into chunks of <= ``MAX_CHUNK`` characters at word boundaries (the model was trained on
   lines up to 75 characters). A normal notebook line is a single chunk.
2. Generate all uncached chunks of a page in one batch, primed with a handwriting style:
   one of the submodule's 13 styles, or "mine" = my own Capture samples.
3. Post-process like the reference: offsets -> coords, Savitzky-Golay denoise, align the
   baseline, flip to +y down.
4. Scale to millimetres from the measured x-height band, then apply slant, letter spacing and
   word spacing, and join the chunks with a word gap.
"""

from __future__ import annotations

import json
import math
import threading
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from hwrobot import strokes as S
from hwrobot.model import hs_bridge
from hwrobot.model.base import Style, word_gap_mm
from hwrobot.model.network import HandwritingRNN

MAX_CHUNK = 70  # the reference accepts lines up to 75 characters
#: Measured from the pretrained model: average character advance / x-height.
DEFAULT_ADVANCE = 0.95
STEPS_PER_CHAR = 40
#: Letter spacing 100% -> +LETTER_GAP x-heights between letters at 200%.
LETTER_GAP = 0.6
SUBSTITUTES = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "`": "'",
    "–": "-", "—": "-", "_": "-", "/": "-",
    "[": "(", "]": ")", "{": "(", "}": ")", "&": "and", "+": "plus",
}


def neatness_to_bias(neatness: float) -> float:
    return 1.5 * min(max(neatness, 0.0), 1.0)


def split_chunks(text: str, limit: int = MAX_CHUNK) -> list[str]:
    chunks, cur = [], ""
    for word in text.split():
        while len(word) > limit:  # a monster word: hard-split it
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(word[:limit])
            word = word[limit:]
        cand = f"{cur} {word}" if cur else word
        if len(cand) > limit and cur:
            chunks.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def xheight_band(y: np.ndarray, frac: float = 0.5) -> tuple[float, float]:
    """Shortest y-interval holding ``frac`` of the points: the body (x-height) zone of a line."""
    ys = np.sort(y)
    n = len(ys)
    k = max(1, int(n * frac))
    if n <= k:
        return float(ys[0]), float(ys[-1])
    widths = ys[k:] - ys[: n - k]
    i = int(np.argmin(widths))
    return float(ys[i]), float(ys[i + k])


class ModelSynth:
    simulated = False

    def __init__(self, weights: str | Path | None = None, samples_dir: str | Path | None = None,
                 net: HandwritingRNN | None = None) -> None:
        self.net = net or HandwritingRNN.from_npz(weights or hs_bridge.weights_path())
        self.d = hs_bridge.drawing()
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self.name = "Handwriting model (pretrained, handwriting-synthesis)"
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self._advance: dict[str, float] = {}
        self._heights: dict[str, list[float]] = {}
        self.last_unsupported: set[str] = set()

    # ------------------------------------------------------------------ styles

    def hands(self) -> list[dict]:
        out = [{"id": f"hs-{i}", "name": f"Style {i}"} for i in hs_bridge.style_ids()]
        if self._my_prime() is not None:
            out.insert(0, {"id": "mine", "name": "Mine (from Capture)"})
        return out

    def _prime(self, hand: str) -> tuple[np.ndarray, str] | None:
        if hand == "mine":
            return self._my_prime()
        if hand.startswith("hs-"):
            try:
                return hs_bridge.load_style(int(hand[3:]))
            except (ValueError, OSError):
                return None
        return None

    def _my_prime(self) -> tuple[np.ndarray, str] | None:
        """Turn my most recent Capture sample into a priming sequence, prepared like IAM data."""
        if not self.samples_dir or not self.samples_dir.exists():
            return None
        files = sorted(self.samples_dir.glob("*.json"))
        for f in reversed(files):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                pts = S.as_array(rec["points"])
            except (OSError, ValueError, KeyError):
                continue
            text, _ = self._prepare(rec.get("prompt", ""))
            if len(pts) < 20 or not text:
                continue
            return self._cache_my_prime(f.name, pts, text)
        return None

    def _cache_my_prime(self, key: str, pts: np.ndarray, text: str) -> tuple[np.ndarray, str]:
        ck = ("prime", key)
        if ck in self._cache:
            return self._cache[ck]  # type: ignore[return-value]
        coords = _resample(pts, target=STEPS_PER_CHAR * 0.6 * len(text))
        coords[:, 1] *= -1  # canvas is +y down; the model expects +y up
        coords = self.d.align(coords)
        coords = self.d.denoise(coords)
        offsets = self.d.coords_to_offsets(coords)
        offsets = self.d.normalize(offsets).astype(np.float32)
        self._cache[ck] = (offsets, text)  # type: ignore[assignment]
        return offsets, text

    # ------------------------------------------------------------------ HandwritingSynth

    def _prepare(self, text: str) -> tuple[str, set[str]]:
        """Map text onto the model's alphabet (``drawing.alphabet``); report what changed."""
        known = set(self.d.alphabet) - {"\x00"}
        out, changed = [], set()
        for ch in text:
            if ch in known:
                out.append(ch)
                continue
            sub = SUBSTITUTES.get(ch)
            if sub is None and ch.isupper() and ch.lower() in known:
                sub = ch.lower()  # the training data has no Q, X or Z capitals
            if sub is None or not all(c in known for c in sub):
                sub = ""
            out.append(sub)
            if not ch.isspace():
                changed.add(ch)
        return " ".join("".join(out).split()), changed

    def measure(self, text: str, style: Style) -> float:
        text, _ = self._prepare(text)
        adv = self._advance.get(style.hand, DEFAULT_ADVANCE) + (style.letter_spacing - 1.0) * LETTER_GAP
        return len(text) * adv * style.x_height_mm

    def write(self, text: str, style: Style) -> S.Points:
        return self.write_many([text], style)[0]

    def write_many(self, texts: list[str], style: Style) -> list[S.Points]:
        prepared = [self._prepare(t) for t in texts]
        self.last_unsupported = set().union(*(c for _, c in prepared)) if prepared else set()
        chunk_lists = [split_chunks(t) for t, _ in prepared]
        bias = round(neatness_to_bias(style.neatness), 3)
        keys = {(c, style.hand, bias, style.seed) for cl in chunk_lists for c in cl}
        with self._lock:
            todo = [k for k in keys if k not in self._cache]
            if todo:
                self._generate(todo, style.hand, bias, style.seed)
            raw = {k: self._cache[k] for k in keys}
            for k in keys:
                self._cache.move_to_end(k)
            while len(self._cache) > 600:
                self._cache.popitem(last=False)
        return [self._assemble(cl, raw, style, bias) for cl in chunk_lists]

    # ------------------------------------------------------------------ generation

    def _generate(self, keys: list[tuple], hand: str, bias: float, seed: int) -> None:
        prime = self._prime(hand)
        for i in range(0, len(keys), 16):
            batch = keys[i:i + 16]
            texts = [k[0] for k in batch]
            prefix = (prime[1] + " ") if prime else ""
            enc = [self.d.encode_ascii(prefix + t) for t in texts]
            U = max(len(e) for e in enc)
            chars = torch.zeros(len(batch), U, dtype=torch.long)
            for r, e in enumerate(enc):
                chars[r, :len(e)] = torch.from_numpy(np.asarray(e, dtype=np.int64))
            lens = torch.tensor([len(e) for e in enc])
            p = pl = None
            if prime is not None:
                p = torch.from_numpy(np.stack([prime[0]] * len(batch))).float()
                pl = torch.full((len(batch),), len(prime[0]))
            g = torch.Generator().manual_seed(zlib.crc32(f"{seed}|{hand}|{'|'.join(texts)}".encode()))
            outs = self.net.sample(chars, lens, torch.full((len(batch),), bias), p, pl,
                                   max_steps=STEPS_PER_CHAR * max(len(t) for t in texts) + 40, generator=g)
            for key, (offsets, idx) in zip(batch, outs):
                idx = idx - len(prefix)
                offsets, idx = _trim_past_end(offsets, idx, len(key[0]))
                coords, h = self._to_line(offsets)
                if len(key[0]) >= 12 and h > 0:
                    self._heights.setdefault(hand, []).append(h)
                self._cache[key] = (coords, idx)

    def _to_line(self, offsets: np.ndarray) -> tuple[np.ndarray, float]:
        """Model offsets -> raw coords with the baseline at y=0, +y down, x from 0; plus band height."""
        if len(offsets) == 0:
            return np.zeros((0, 3)), 0.0
        offsets = offsets.copy()
        offsets[-1, 2] = 1.0
        coords = self.d.offsets_to_coords(offsets)
        coords = self.d.denoise(coords)
        coords[:, :2] = self.d.align(coords[:, :2])
        coords[:, 1] *= -1
        coords, h = straighten(coords)
        coords[:, 0] -= coords[:, 0].min()
        return coords, h

    def _unit(self, hand: str) -> float:
        """Model units per x-height for this hand: the median over every long chunk seen so far."""
        hs = self._heights.get(hand)
        return float(np.median(hs[-200:])) if hs else 1.0

    # ------------------------------------------------------------------ assembly

    def _assemble(self, chunks: list[str], raw: dict, style: Style, bias: float) -> S.Points:
        xh = style.x_height_mm
        gap = word_gap_mm(style)
        extra_word = (style.word_spacing - 1.0) * 1.1 * xh
        shear = math.tan(math.radians(style.slant_deg))
        parts, x = [], 0.0
        unit = self._unit(style.hand)
        for chunk in chunks:
            coords, idx = raw[(chunk, style.hand, bias, style.seed)]
            if len(coords) == 0:
                continue
            pts = coords.copy()
            pts[:, :2] *= xh / unit
            if style.letter_spacing != 1.0:
                pts = _track_letters(pts, idx, (style.letter_spacing - 1.0) * LETTER_GAP * xh)
            pts[:, 0] -= pts[:, 1] * shear
            if extra_word:
                pts = _spread_words(pts, idx, chunk, extra_word)
            x0 = pts[:, 0].min()
            pts[:, 0] += x - x0
            parts.append(pts)
            x = pts[:, 0].max() + gap
            natural = pts[:, 0].max() - pts[:, 0].min() - (style.letter_spacing - 1.0) * LETTER_GAP * xh * len(chunk)
            self._learn_advance(style.hand, chunk, natural / xh)
        if not parts:
            return np.zeros((0, 3))
        return S.as_array(np.vstack(parts))

    def _learn_advance(self, hand: str, chunk: str, width_units: float) -> None:
        if len(chunk) < 4:
            return
        a = width_units / len(chunk)
        old = self._advance.get(hand, DEFAULT_ADVANCE)
        self._advance[hand] = 0.8 * old + 0.2 * a


def baseline_points(coords: np.ndarray, h: float) -> np.ndarray:
    """Bottoms of letter bodies: each stroke's local lowest points (+y down) near their median.

    Most low points of handwriting are letters touching the baseline; descender tails and the
    bottoms of high loops are a minority, so keeping only lows within ~half an x-height of the
    median stops 'y', 'g', 'f' and tall letters pulling the fit around.
    """
    lows = []
    rise, w = 0.2 * h, 10
    for s in S.split_strokes(coords):
        n = len(s)
        if n < 3:
            continue
        y = s[:, 1]
        for i in range(n):
            left, right = y[max(0, i - w):i], y[i + 1:i + 1 + w]
            near = y[max(0, i - 3):i + 4]
            if y[i] < near.max():
                continue  # not the lowest point around here
            up_left = len(left) and y[i] - left.min() > rise
            up_right = len(right) and y[i] - right.min() > rise
            if (up_left and up_right) or (up_left and i >= n - 3) or (up_right and i <= 2):
                lows.append(s[i, :2])
    lows = np.array(lows).reshape(-1, 2)
    if len(lows) == 0:
        return lows
    med = np.median(lows[:, 1])
    return lows[np.abs(lows[:, 1] - med) <= 0.5 * h]


def straighten(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Level the line on its real baseline and put that baseline at y = 0 (+y down).

    Robust line fit through :func:`baseline_points` (two passes, dropping outliers), rotate it
    flat, and sit the median letter bottom on y = 0. Returns the coords and the x-height band
    height (model units). Any wobble left is natural; per-line nudges in the UI handle the rest.
    """
    c = coords.copy()
    top, base = xheight_band(c[:, 1])
    h = max(base - top, 1e-6)
    for _ in range(2):  # a second pass catches letters the first window missed on a tilted line
        pts = baseline_points(c, h)
        if len(pts) < 4:
            break
        keep = np.ones(len(pts), bool)
        for _ in range(2):
            b, a = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
            resid = pts[:, 1] - (a + b * pts[:, 0])
            keep = np.abs(resid) < 0.3 * h
            if keep.sum() < 4:
                keep[:] = True
                break
        b, a = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
        theta = -math.atan(b)
        x0 = c[:, 0].min()
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        c[:, :2] = (c[:, :2] - [x0, a + b * x0]) @ rot.T
    pts = baseline_points(c, h)
    c[:, 1] -= float(np.median(pts[:, 1])) if len(pts) >= 3 else base
    return c, h


def _trim_past_end(offsets: np.ndarray, idx: np.ndarray, n_chars: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop strokes that start after the attention has moved past the last character."""
    if len(offsets) == 0:
        return offsets, idx
    keep_until = len(offsets)
    start = 0
    for end in np.flatnonzero(offsets[:, 2] == 1) + 1:
        if start > 0 and idx[start] >= n_chars:
            keep_until = start
            break
        start = end
    return offsets[:keep_until], idx[:keep_until]


def _track_letters(pts: np.ndarray, idx: np.ndarray, extra: float) -> np.ndarray:
    """Add ``extra`` mm between letters (negative tightens).

    Each point moves right by (letter position x extra), where the letter position is the
    model's attention index, smoothed so a joining stroke stretches gradually instead of
    jumping. Letter shapes stay intact; the connectors between them lengthen.
    """
    if len(idx) != len(pts) or len(pts) < 2:
        return pts
    pos = idx.astype(float)
    k = min(9, len(pos) | 1)
    pad = k // 2
    pos = np.convolve(np.pad(pos, pad, mode="edge"), np.ones(k) / k, mode="valid")
    pos = np.maximum.accumulate(pos)  # never step backwards
    # i-dots and t-crosses are often added after the word, when attention has moved on:
    # give each short stroke the position of the letter body it sits over instead.
    ends = np.flatnonzero(pts[:, 2] == 1) + 1
    starts = np.r_[0, ends[:-1]]
    big = np.zeros(len(pts), bool)
    for s, e in zip(starts, ends):
        big[s:e] = e - s > 12
    if big.any():
        bx = pts[big, 0]
        bpos = pos[big]
        for s, e in zip(starts, ends):
            if e - s <= 12:
                cx = pts[s:e, 0].mean()
                pos[s:e] = bpos[np.argmin(np.abs(bx - cx))]
    out = pts.copy()
    out[:, 0] += pos * extra
    return out


def _spread_words(pts: np.ndarray, idx: np.ndarray, text: str, extra: float) -> np.ndarray:
    """Push word k right by ``k * extra`` mm, deciding each stroke's word by where it starts."""
    if len(idx) != len(pts):
        return pts
    spaces = np.cumsum([c == " " for c in text])
    out = pts.copy()
    start = 0
    for end in np.flatnonzero(pts[:, 2] == 1) + 1:
        i = int(np.clip(idx[start], 0, len(text) - 1))
        out[start:end, 0] += spaces[i] * extra
        start = end
    return out


def _resample(pts: np.ndarray, target: float) -> np.ndarray:
    """Resample strokes along arc length so the total point count is about ``target``."""
    strokes = S.split_strokes(pts)
    lengths = [float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in strokes]
    total = sum(lengths) or 1.0
    out = []
    for s, L in zip(strokes, lengths):
        n = max(2, int(round(target * L / total))) if len(s) > 1 else 1
        if len(s) == 1:
            xy = s
        else:
            d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))])
            t = np.linspace(0, d[-1], n)
            xy = np.column_stack([np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])])
        eos = np.zeros((len(xy), 1))
        eos[-1] = 1
        out.append(np.hstack([xy, eos]))
    return np.vstack(out)
