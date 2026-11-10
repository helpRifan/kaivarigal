# Handwriting Robot (`hwrobot`)

A pen plotter that writes on real paper in my own handwriting.

```
typed text → handwriting model → strokes (mm) → layout onto detected lines → G-code → GRBL → paper
```

See `CLAUDE.md` for the full design. **Current phase: 1 — skeleton.**

## Setup (Windows, PowerShell)

Requires Python 3.11+ (`py --list` shows what's installed).

```powershell
cd E:\handwrit
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"), run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(In Git Bash use `source .venv/Scripts/activate`; in cmd.exe use `.venv\Scripts\activate.bat`.)

## Run the tests

```powershell
pytest
```

## Try the simulator

```powershell
python -m hwrobot.plotter out\demo.png
```

Renders a small demo stroke set onto an A4 page and writes `out\demo.png` (plus `out\demo.svg`).

## Layout

| Path | What |
|---|---|
| `src/hwrobot/strokes.py` | Shared stroke format + all conversions |
| `src/hwrobot/plotter/` | Plotter interface, simulator (PNG/SVG), G-code + GRBL (later) |
| `src/hwrobot/capture/` | Handwriting sample capture (Phase 2) |
| `src/hwrobot/model/` | Handwriting generation model (Phase 2) |
| `src/hwrobot/vision/` | Page + ruled-line detection (Phase 3) |
| `src/hwrobot/layout/` | Fit strokes onto lines (Phase 3) |
| `src/hwrobot/calibration/` | Camera px ↔ plotter mm (Phase 5) |
| `src/hwrobot/app/` | FastAPI web UI (Phase 4) |

## References

- A. Graves, *Generating Sequences With Recurrent Neural Networks*, arXiv:1308.0850
- sjvasquez/handwriting-synthesis — a TensorFlow implementation of the Graves model (design reference)
