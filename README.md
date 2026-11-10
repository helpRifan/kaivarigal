# Handwriting Robot (`hwrobot`)

A pen plotter that writes on real paper in my own handwriting.

```
typed text → handwriting model → strokes (mm) → layout onto detected lines → G-code → GRBL → paper
```

See `CLAUDE.md` for the full design and `docs/roadmap.md` for the build order.
**Current state:** the web app runs end to end with **real handwriting** from a pretrained model
(13 reference styles, or my own once I've saved Capture samples). Page vision is still faked
(guessed lines) and plotting is simulated.

## Setup (Windows, PowerShell)

Requires Python 3.11+ (`py --list` shows what's installed).

```powershell
git clone --recurse-submodules https://github.com/helpRifan/kaivarigal.git
cd kaivarigal
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,app,model]"
```

Already cloned without the submodule? `git submodule update --init`.

### Handwriting model weights (one time)

The pretrained model comes from the `third_party/handwriting-synthesis` submodule as a
TensorFlow 1 checkpoint. Convert it once into `models/hs_weights.npz` (git-ignored); TensorFlow
is only needed for this step, in its own throwaway environment:

```powershell
py -3.12 -m venv .tfenv
.\.tfenv\Scripts\pip install tensorflow-cpu numpy
.\.tfenv\Scripts\python tools\convert_hs_checkpoint.py
```

Without the weights the app still runs, using placeholder scribble.

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"), run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(In Git Bash use `source .venv/Scripts/activate`; in cmd.exe use `.venv\Scripts\activate.bat`.)

## Run the tests

```powershell
pytest
```

## Run the app

```powershell
python -m hwrobot.app
```

Open http://localhost:8000. To use it from your phone on the same Wi-Fi, start it with
`python -m hwrobot.app --host 0.0.0.0` and open `http://<laptop-ip>:8000` on the phone.
Samples, history and settings are saved under `data\` (git-ignored).

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

## Credits

- **[sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)** by
  Sean Vasquez: the handwriting model this project runs. Used via my fork
  ([helpRifan/handwriting-synthesis](https://github.com/helpRifan/handwriting-synthesis)) as the
  `third_party/handwriting-synthesis` submodule: its pretrained checkpoint, its 13 priming styles
  and its `drawing.py` preprocessing. `src/hwrobot/model/network.py` is a PyTorch port of its
  TensorFlow network so the same weights run on current Python and on a Raspberry Pi.
- A. Graves, *Generating Sequences With Recurrent Neural Networks*,
  [arXiv:1308.0850](https://arxiv.org/abs/1308.0850): the method.
- The pretrained model was trained on the
  [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database).
