# CLAUDE.md — Handwriting Robot

## What this project is

A robot that writes on real paper in **my own handwriting**.

1. **Handwriting AI** — learns my handwriting from samples, turns typed text into pen strokes that look human (natural variation, not a font).
2. **Page vision** — camera photo of a notebook page → flatten it → detect the ruled lines → I pick which line gets which text.
3. **Plotter** — a 2-axis pen plotter (X/Y steppers + servo pen lift) writes the strokes onto the page.

Pipeline:
```
typed text → handwriting model → strokes (mm) → layout onto detected lines → G-code → GRBL → paper
```

## Hardware status

**No hardware yet.** Everything must run and be testable on the laptop alone.

Planned hardware (later):
- Raspberry Pi 4 (2GB+) — runs the app, model, and camera
- Pi Camera Module v2 (laptop webcam/phone photos until then)
- Arduino Uno + CNC shield + 2× A4988 + 2× NEMA17 steppers + SG90 servo, running **GRBL** (servo-pen fork)
- Pi ↔ Arduino over USB serial; Pi serves a web UI over Wi-Fi so I can control it from my phone

Because of this, the plotter is behind an interface with a **simulator backend** (renders strokes to an image/SVG) and a **GRBL serial backend** (added later). Never make code depend on hardware being present.

## Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.11+ |
| Handwriting model | PyTorch — start with Graves-style LSTM + attention + mixture density output (arXiv 1308.0850); pretrain on IAM-OnDB, fine-tune on my samples |
| Vision | OpenCV + NumPy |
| Backend / UI | FastAPI + simple web frontend (works from phone browser) |
| Plotter comms | pyserial → GRBL |
| Tests | pytest |
| Packaging | `pyproject.toml`, venv |

Keep the model small enough to run inference on a Pi 4 (CPU). Training happens on the laptop / Colab.

## Project structure

```
handwriting-robot/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/hwrobot/
│   ├── strokes.py          # Stroke data types + conversions (shared format, see below)
│   ├── capture/            # Collecting my handwriting samples
│   ├── model/              # Handwriting generation model: dataset, train, sample
│   ├── vision/             # Page detection, perspective warp, ruled-line detection
│   ├── layout/             # Fit generated strokes onto detected lines (wrap, scale, margins)
│   ├── plotter/
│   │   ├── base.py         # Plotter interface
│   │   ├── simulator.py    # Renders to PNG/SVG — default backend
│   │   ├── gcode.py        # Strokes → G-code (pen up/down, feed rate)
│   │   └── grbl.py         # Serial streaming to GRBL (later)
│   ├── calibration/        # Camera-pixel ↔ plotter-mm mapping
│   └── app/                # FastAPI server + web UI
├── data/                   # Samples, datasets (git-ignored except small fixtures)
├── models/                 # Checkpoints (git-ignored)
├── notebooks/              # Experiments
└── tests/
```

## Shared stroke format (the contract between modules)

- A handwriting sample = list of points `(x, y, pen_up)`; `pen_up=1` means the stroke ends after this point.
- Model works in **relative offsets** `(dx, dy, pen_up)` (IAM-OnDB style), normalized.
- Everything after the model works in **absolute millimetres on the page**, origin top-left, +y down.
- `strokes.py` owns all conversions. Other modules must not invent their own format.

## Software features (build order)

**Core**
1. Handwriting capture — draw on a canvas in the browser (mouse/stylus/phone touch) to record real stroke timing; prompt me with sentences covering all letters.
2. Model — generate text in my style; "neatness" (bias) and slant/size controls.
3. Page scan — detect page corners, warp flat, detect ruled lines + margin.
4. Line assignment UI — click a line, type text; auto-wrap long text across lines.
5. Preview — overlay generated handwriting on the scanned page before writing.
6. Plot — simulator now, GRBL later.

**Extras (after core works)**
- Since not only every letter of my handwriting it identifies, but the hgiht of  the next letter also differs right 
- Style profiles (normal / exam / quick notes)
- Partial page fill — detect lines that already have writing and skip them
- Templates (lab record, assignment page) so no photo is needed every time
- Import text from .txt / .docx, overflow warning
- Speed control (feed rate)
- Calibration routine — plot a test grid, photograph it, auto-correct skew/offset
- Job history

## Phases

1. **Skeleton** — repo, pyproject, module stubs with docstrings, stroke format, simulator plotter, pytest passing. ← *start here*
2. **Capture + model** — capture tool, IAM-OnDB pretraining, fine-tune on my samples.
3. **Vision + layout** — page/line detection on phone photos, layout engine, preview.
4. **App** — FastAPI web UI tying it together, simulator output.
5. **Hardware** — G-code + GRBL backend, calibration, move to the Pi.


## Honest-use note

This is a personal robotics + ML project. The robot writes my own text in my own handwriting.
