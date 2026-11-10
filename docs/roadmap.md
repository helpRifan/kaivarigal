# Roadmap

Order agreed on 2026-11-11: build the **whole app first with fake data**, then replace each fake
with the real thing, one phase at a time. Every fake sits behind the same interface the real part
will use, so swapping one in never changes the UI.

| Step | What | Status |
|---|---|---|
| 1 | Skeleton: stroke format, simulator plotter, tests | done |
| 2 | **Web app with fake data**: every screen working end to end | done |
| 4a | Handwriting model: PyTorch port of handwriting-synthesis running its pretrained weights; 13 styles + priming with my Capture samples | done |
| 3 | Page vision: real corner detection, perspective warp, ruled-line detection (OpenCV) | next |
| 4b | Fine-tune the model on my Capture samples (needs a GPU / Colab) | after enough samples |
| 5 | Hardware: G-code, GRBL serial backend, real calibration, move to the Pi | when parts arrive |

## The app (step 2)

One web app (FastAPI + plain HTML/CSS/JS, no build step, no internet needed):

- **Windows now:** `python -m hwrobot.app` and open `http://localhost:8000`.
- **Pi later:** the same server, Chromium in kiosk mode on the LCD. The layout works down to
  800×480 with touch-sized controls and nothing that needs hover.
- **Phone:** the same page over Wi-Fi.

Screens:

| Screen | Does | Fake today |
|---|---|---|
| Write | Page with ruled lines; tap a line, type; text wraps onto the next empty lines; style sliders; live preview over the page; plot with speed, trace-area, pause, stop, progress | Handwriting is a placeholder scribble generator, not my handwriting |
| Scan | Webcam photo, photo upload, or a page template | Line detection on photos is guessed (evenly spaced lines) |
| Capture | Write prompted sentences on a canvas; samples saved in the shared stroke format | Nothing fake: this is real and is the training data for step 4 |
| History | Every job: when, which page, what text, the finished page image | Real |
| Calibrate | Plot a test grid, photograph it, apply the correction; machine settings | Measurement numbers are made up |

Fakes and the real modules that replace them:

| Interface | Fake now | Real later |
|---|---|---|
| `hwrobot.model.base.HandwritingSynth` | `hwrobot.model.fake.FakeSynth` | the trained model (step 4) |
| `hwrobot.vision` page source | `hwrobot.vision.fake` | OpenCV pipeline (step 3) |
| `hwrobot.plotter.base.Plotter` | `SimulatorPlotter` | `grbl.py` (step 5) |

## Feature order (from my list, with Claude's additions)

First: preview overlay, style sliders + profiles, text import with auto-wrap and overflow warning,
calibration, capture screen, trace-area, pause/stop, progress.
Then: templates, partial page fill (skip written lines), speed control, history.
Later or never: pen colour detection.
