"""FastAPI server: JSON API + the static web UI.

Run with ``python -m hwrobot.app`` (see ``__main__.py``). All state is in memory except
history, settings and capture samples, which live under the data folder.
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageChops
from pydantic import BaseModel, Field

from hwrobot import strokes as S
from hwrobot.layout.engine import LayoutResult, layout_page
from hwrobot.model.base import PRESETS, Style
from hwrobot.model.fake import FakeSynth
from hwrobot.page import Page
from hwrobot.plotter.simulator import SimulatorPlotter
from hwrobot.vision import fake as vision

STATIC = Path(__file__).parent / "static"

CAPTURE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
    "Jackdaws love my big sphinx of quartz.",
    "The five boxing wizards jump quickly.",
    "Waltz, bad nymph, for quick jigs vex.",
    "Bright vixens jump; dozy fowl quack.",
    "My lab record is due on Monday, 12 March.",
    "Write the answer in about 120 words.",
    "Glib jocks quiz nymph to vex dwarf.",
    "Amazingly few discotheques provide jukeboxes.",
]

DEFAULT_SETTINGS = {"pen_width_mm": 0.5, "feed_rate": 1500, "sim_speedup": 12.0,
                    "calibration": {"skew_deg": 0.0, "offset_x_mm": 0.0, "offset_y_mm": 0.0, "scale": 1.0}}


# ----------------------------------------------------------------------------- request models


class StyleIn(BaseModel):
    x_height_mm: float = Field(2.6, ge=1.2, le=6.0)
    slant_deg: float = Field(8.0, ge=-20, le=35)
    neatness: float = Field(0.7, ge=0, le=1)
    letter_spacing: float = Field(1.0, ge=0.6, le=1.6)
    word_spacing: float = Field(1.0, ge=0.5, le=2.0)
    seed: int = 0

    def to_style(self) -> Style:
        return Style(**self.model_dump())


class WriteIn(BaseModel):
    page_id: str
    assignments: dict[str, str] = {}
    style: StyleIn = StyleIn()
    skip_used: bool = True


class JobIn(WriteIn):
    feed_rate: float | None = Field(None, ge=200, le=6000)
    trace_only: bool = False


class TemplateIn(BaseModel):
    template_id: str


class SampleIn(BaseModel):
    prompt: str
    points: list[list[float]]
    times_ms: list[float] = []
    canvas_px: list[float] = []


# ----------------------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    created: str
    page: Page
    label: str
    text_preview: str
    points: S.Points
    feed_rate: float
    trace_only: bool
    total_mm: float
    drawn_mm: float = 0.0
    state: str = "running"  # running | paused | done | stopped
    image_path: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def summary(self) -> dict[str, Any]:
        remaining = max(self.total_mm - self.drawn_mm, 0.0)
        return {
            "id": self.id,
            "created": self.created,
            "label": self.label,
            "page_id": self.page.id,
            "text_preview": self.text_preview,
            "state": self.state,
            "trace_only": self.trace_only,
            "feed_rate": self.feed_rate,
            "drawn_mm": round(self.drawn_mm, 1),
            "total_mm": round(self.total_mm, 1),
            "progress": 1.0 if self.total_mm == 0 else min(self.drawn_mm / self.total_mm, 1.0),
            "eta_s": round(remaining / self.feed_rate * 60.0, 1),
            "image_url": f"/api/jobs/{self.id}/image.png" if self.image_path else None,
        }


def _path_length(points: S.Points) -> float:
    return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in S.split_strokes(points)))


def _svg_paths(points: S.Points) -> list[str]:
    out = []
    for s in S.split_strokes(points):
        if len(s) == 1:
            s = np.vstack([s, s])
        out.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in s))
    return out


def _trace_points(points: S.Points) -> S.Points:
    x0, y0, x1, y1 = S.bounding_box(points)
    return S.from_strokes([[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])


def _grid_points(page: Page, step: float = 20.0) -> S.Points:
    strokes = []
    x0, y0, x1, y1 = 20.0, 20.0, page.width_mm - 20.0, page.height_mm - 20.0
    for x in np.arange(x0, x1 + 0.01, step):
        strokes.append([(x, y0), (x, y1)])
    for y in np.arange(y0, y1 + 0.01, step):
        strokes.append([(x0, y), (x1, y)])
    return S.from_strokes(strokes)


# ----------------------------------------------------------------------------- app


class AppState:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.synth = FakeSynth()
        self.pages: dict[str, Page] = {}
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.settings = self._load_json("settings.json", DEFAULT_SETTINGS)
        self.history: list[dict] = self._load_json("history.json", [])

    # -- persistence
    def _load_json(self, name: str, default: Any) -> Any:
        path = self.data_dir / name
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(default, dict):
                    return {**default, **loaded}
                return loaded
            except (OSError, ValueError):
                pass
        return json.loads(json.dumps(default))

    def save_json(self, name: str, value: Any) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    def sample_count(self) -> int:
        return len(list(self.samples_dir.glob("*.json"))) if self.samples_dir.exists() else 0

    # -- pages
    def add_page(self, page: Page) -> Page:
        with self.lock:
            self.pages[page.id] = page
        return page

    def get_page(self, page_id: str) -> Page:
        page = self.pages.get(page_id)
        if page is None:
            raise HTTPException(404, "Page not found. Scan a page or pick a template first.")
        return page

    def layout(self, req: WriteIn) -> tuple[Page, LayoutResult]:
        page = self.get_page(req.page_id)
        try:
            return page, layout_page(page, req.assignments, self.synth, req.style.to_style(), req.skip_used)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- jobs
    def active_job(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.state in ("running", "paused")), None)

    def start_job(self, page: Page, points: S.Points, label: str, preview: str,
                  feed_rate: float | None, trace_only: bool) -> Job:
        with self.lock:
            if self.active_job():
                raise HTTPException(409, "The plotter is already busy. Stop or finish the current job first.")
            if len(points) == 0:
                raise HTTPException(400, "Nothing to write yet. Type some text on a line.")
            job = Job(
                id=uuid.uuid4().hex[:8],
                created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                page=page, label=label, text_preview=preview, points=points,
                feed_rate=float(feed_rate or self.settings["feed_rate"]), trace_only=trace_only,
                total_mm=_path_length(points),
            )
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        last = time.monotonic()
        while True:
            time.sleep(0.05)
            now = time.monotonic()
            dt, last = now - last, now
            with job.lock:
                if job.state == "paused":
                    continue
                if job.state != "running":
                    break
                job.drawn_mm += job.feed_rate / 60.0 * dt * float(self.settings["sim_speedup"])
                if job.drawn_mm >= job.total_mm:
                    job.drawn_mm = job.total_mm
                    job.state = "done"
                    break
        self._finish(job)

    def _finish(self, job: Job) -> None:
        if not job.trace_only:
            job.image_path = str(self._render_job(job))
        if not job.trace_only:
            entry = job.summary()
            self.history.insert(0, entry)
            del self.history[200:]
            self.save_json("history.json", self.history)

    def _render_job(self, job: Job) -> Path:
        sim = SimulatorPlotter(page_size_mm=(job.page.width_mm, job.page.height_mm),
                               px_per_mm=vision.PX_PER_MM, pen_width_mm=float(self.settings["pen_width_mm"]))
        pts = job.points
        if job.state == "stopped":
            pts = _truncate(pts, job.drawn_mm)
        if len(pts):
            sim.plot(pts)
        ink = sim.render()
        out = ink
        if job.page.image is not None:
            out = ImageChops.multiply(job.page.image.resize(ink.size), ink)
        path = self.data_dir / "jobs" / f"{job.id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path)
        return path


def _truncate(points: S.Points, length_mm: float) -> S.Points:
    """Keep only the first ``length_mm`` of pen-down drawing (what a stopped job actually drew)."""
    kept, left = [], length_mm
    for s in S.split_strokes(points):
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        total = float(seg.sum())
        if total <= left:
            kept.append(s)
            left -= total
            continue
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = int(np.searchsorted(cum, left, side="right"))
        if n >= 1:
            kept.append(s[:n])
        break
    return S.from_strokes(kept)


def create_app(data_dir: str | Path = "data") -> FastAPI:
    state = AppState(Path(data_dir))
    app = FastAPI(title="Kaivarigal", docs_url="/api/docs", redoc_url=None)
    app.state.hw = state

    # -- status / settings
    @app.get("/api/status")
    def status() -> dict:
        job = state.active_job()
        return {
            "backend": "simulator",
            "robot_connected": False,
            "model": state.synth.name,
            "model_simulated": state.synth.simulated,
            "vision_simulated": True,
            "job": job.summary() if job else None,
            "samples": state.sample_count(),
        }

    @app.get("/api/settings")
    def get_settings() -> dict:
        return state.settings

    @app.put("/api/settings")
    def put_settings(body: dict) -> dict:
        allowed = {"pen_width_mm": (0.1, 2.0), "feed_rate": (200, 6000), "sim_speedup": (1, 1e7)}
        for key, (lo, hi) in allowed.items():
            if key in body:
                value = float(body[key])
                if not lo <= value <= hi:
                    raise HTTPException(400, f"{key} must be between {lo:g} and {hi:g}")
                state.settings[key] = value
        state.save_json("settings.json", state.settings)
        return state.settings

    # -- pages
    @app.get("/api/templates")
    def templates() -> list[dict]:
        return [{"id": t.id, "name": t.name, "description": t.description,
                 "thumb_url": f"/api/templates/{t.id}/thumb.png"} for t in vision.TEMPLATES.values()]

    _thumbs: dict[str, bytes] = {}

    @app.get("/api/templates/{template_id}/thumb.png")
    def template_thumb(template_id: str) -> Response:
        if template_id not in vision.TEMPLATES:
            raise HTTPException(404, "No such template")
        if template_id not in _thumbs:
            img = vision.make_page(template_id).image
            img.thumbnail((240, 340))
            _thumbs[template_id] = _png(img)
        return Response(_thumbs[template_id], media_type="image/png")

    @app.post("/api/pages/from-template")
    def page_from_template(body: TemplateIn) -> dict:
        if body.template_id not in vision.TEMPLATES:
            raise HTTPException(404, "No such template")
        return _page_json(state.add_page(vision.make_page(body.template_id)))

    @app.post("/api/pages/from-photo")
    async def page_from_photo(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "Photo is larger than 25 MB.")
        try:
            page = vision.scan_photo(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _page_json(state.add_page(page))

    @app.get("/api/pages/{page_id}")
    def get_page(page_id: str) -> dict:
        return _page_json(state.get_page(page_id))

    @app.get("/api/pages/{page_id}/image.png")
    def page_image(page_id: str) -> Response:
        page = state.get_page(page_id)
        if page.image is None:
            raise HTTPException(404, "Page has no image")
        return Response(_png(page.image), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    # -- writing
    @app.get("/api/styles")
    def styles() -> list[dict]:
        return [{"id": k, "name": name, "style": s.to_dict()} for k, (name, s) in PRESETS.items()]

    @app.post("/api/preview")
    def preview(body: WriteIn) -> dict:
        _, result = state.layout(body)
        pts = result.all_points()
        return {
            "lines": [{"line_id": l.line_id, "text": l.text, "continued": l.continued,
                       "paths": _svg_paths(l.points)} for l in result.lines],
            "overflow": result.overflow,
            "warnings": result.warnings,
            "ink_mm": round(_path_length(pts), 1) if len(pts) else 0.0,
            "bbox": S.bounding_box(pts) if len(pts) else None,
        }

    # -- jobs
    @app.post("/api/jobs")
    def create_job(body: JobIn) -> dict:
        page, result = state.layout(body)
        pts = result.all_points()
        if body.trace_only and len(pts):
            pts = _trace_points(pts)
        text = " ".join(l.text for l in result.lines)
        job = state.start_job(page, pts, page.label, text[:80], body.feed_rate, body.trace_only)
        return job.summary()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return state.history

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        return job.summary()

    @app.post("/api/jobs/{job_id}/{action}")
    def job_action(job_id: str, action: str) -> dict:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        transitions = {"pause": ("running", "paused"), "resume": ("paused", "running"),
                       "stop": (None, "stopped")}
        if action not in transitions:
            raise HTTPException(400, "Action must be pause, resume or stop")
        need, to = transitions[action]
        with job.lock:
            if job.state in ("done", "stopped"):
                raise HTTPException(409, f"Job is already {job.state}")
            if need and job.state != need:
                raise HTTPException(409, f"Can't {action} a {job.state} job")
            job.state = to
        return job.summary()

    @app.get("/api/jobs/{job_id}/image.png")
    def job_image(job_id: str) -> FileResponse:
        path = state.data_dir / "jobs" / f"{job_id}.png"
        if not path.exists():
            raise HTTPException(404, "No image for this job yet")
        return FileResponse(path, media_type="image/png")

    # -- capture
    @app.get("/api/capture/prompts")
    def prompts() -> dict:
        return {"prompts": CAPTURE_PROMPTS, "samples": state.sample_count(), "goal": 60}

    @app.post("/api/capture/samples")
    def save_sample(body: SampleIn) -> dict:
        try:
            pts = S.as_array(body.points)
        except ValueError as exc:
            raise HTTPException(400, f"Bad stroke data: {exc}") from exc
        if len(pts) < 2:
            raise HTTPException(400, "Write the sentence on the canvas first.")
        if body.times_ms and len(body.times_ms) != len(pts):
            raise HTTPException(400, "times_ms must have one entry per point")
        state.samples_dir.mkdir(parents=True, exist_ok=True)
        sample_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        record = {
            "id": sample_id, "prompt": body.prompt, "units": "px", "canvas_px": body.canvas_px,
            "points": pts.tolist(), "times_ms": body.times_ms,
        }
        (state.samples_dir / f"{sample_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return {"id": sample_id, "samples": state.sample_count()}

    # -- calibration
    @app.post("/api/calibration/grid")
    def calibration_grid() -> dict:
        page = state.add_page(vision.blank_page())
        pts = _grid_points(page)
        return state.start_job(page, pts, "Calibration grid", "Test grid, 20 mm squares", None, False).summary()

    @app.post("/api/calibration/measure")
    async def calibration_measure(file: UploadFile | None = File(None)) -> dict:
        # FAKE: real version detects the grid in the photo and fits an affine transform.
        rng = np.random.default_rng()
        return {"simulated": True, "skew_deg": round(float(rng.normal(0, 0.4)), 2),
                "offset_x_mm": round(float(rng.normal(0, 0.6)), 2),
                "offset_y_mm": round(float(rng.normal(0, 0.6)), 2),
                "scale": round(float(1 + rng.normal(0, 0.004)), 4)}

    @app.post("/api/calibration/apply")
    def calibration_apply(body: dict) -> dict:
        keys = ("skew_deg", "offset_x_mm", "offset_y_mm", "scale")
        state.settings["calibration"] = {k: float(body.get(k, DEFAULT_SETTINGS["calibration"][k])) for k in keys}
        state.save_json("settings.json", state.settings)
        return state.settings

    # -- UI (always revalidate, so an updated UI shows up without a hard refresh)
    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    return app


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _page_json(page: Page) -> dict:
    return {**page.to_dict(), "image_url": f"/api/pages/{page.id}/image.png"}
