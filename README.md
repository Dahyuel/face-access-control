# Biometric Face Recognition — Access Control System

A biometric access-control system that implements and benchmarks two classical face recognition algorithms — **PCA (Eigenfaces)** and **LBP (Local Binary Patterns)** — with a Flask web dashboard, live webcam enrollment/verification, and a full biometric evaluation suite.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## Features

- **Two recognition engines** — PCA/Eigenfaces (holistic, appearance-based) and LBP (local, texture-based, illumination-robust).
- **Full biometric evaluation** — EER, AUC, d-prime, Rank-1 accuracy, TMR @ FMR=1% / 0.01%, FPIR and FNIR @ EER threshold.
- **Live enrollment & verification** — capture face images from a webcam, enroll new users, and perform 1:N identification.
- **Multi-frame robust verification** — majority-vote identity + median score across frames to resist single-frame outliers.
- **MediaPipe face detection & alignment** — landmark-based cropping, head-pose checks, and optional liveness (blink detection).
- **Auto-calibrated thresholds** — verification and duplicate thresholds written to `calibration.json` by the pipeline.
- **Dark glassmorphism dashboard** — vanilla HTML/CSS/JS single-page app with live overlays and generated result plots.
- **No deep-learning dependency** — runs entirely on classical CV/ML (OpenCV, NumPy, scikit-learn, MediaPipe).

---

## Prerequisites

- Python 3.9 or newer
- A working webcam (for enrollment and live verification)
- pip / venv

---

## Installation

```bash
git clone https://github.com/Hagras99/AccessControl_Face-Recognition-system.git
cd AccessControl_Face-Recognition-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If `requirements.txt` is not present, install the core dependencies directly:

```bash
pip install flask opencv-python numpy scikit-learn mediapipe matplotlib
```

> The MediaPipe face landmark model (`models/face_landmarker.task`) ships with the repo. Keep it in place.

---

## Usage

### Run the web dashboard

```bash
python app.py
```

Then open <http://localhost:5000> in your browser.

| Page action | Endpoint | Description |
|---|---|---|
| Start Evaluation | `POST /api/run` | Runs the PCA/LBP pipeline on the ORL dataset |
| Enroll | `POST /api/enroll` | Enrolls a user from base64 face images |
| Verify | `POST /api/verify` | 1:N identification (single or multi-frame) |
| Detect | `POST /api/detect` | Live bounding-box detection for the overlay |
| Users | `GET /api/users` | Lists enrolled users |
| Delete | `DELETE /api/users/<id>` | Removes an enrolled user |

### Run the evaluation pipeline directly (CLI)

```bash
python main.py
```

This loads the ORL dataset, evaluates PCA and LBP, prints the metric table, and writes `output/final_dashboard.png` (also copied to `static/images/results.png`).

---

## Project Structure

```
.
├── app.py                  # Flask REST API + SPA host
├── main.py                 # Core ML pipeline (PCA/LBP, metrics, plots)
├── enrollment.py           # Enrollment, detection, verification, liveness
├── calibration.json        # Auto-generated verification thresholds
├── users.json              # Enrolled subject registry
├── models/
│   └── face_landmarker.task
├── dataset/                # ORL dataset — s1 … s40 (10 PGM images each)
├── live_dataset/           # Captured webcam enrollment images
├── output/                 # Generated dashboard PNG
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/results.png
├── templates/
│   └── index.html
└── REPORT.md               # Detailed technical report
```

---

## How It Works

### Preprocessing

Each face image is resized to `100×100` and histogram-equalised to reduce illumination sensitivity before feature extraction.

### PCA — Eigenfaces

Images are flattened to 10,000-D vectors; PCA is fit **only on the enroll set** (no data leakage), retaining the top `min(50, n_enroll − 1)` components. Enroll and probe images are projected into this subspace and compared with cosine similarity.

### LBP — Local Binary Patterns

A NumPy-vectorised LBP operator computes an 8-bit code per pixel; a 5×5 spatial grid of uniform-LBP histograms produces a 1475-D feature vector. Relative pixel comparisons make LBP inherently illumination-robust.

### Matching

Both methods use cosine similarity, numerically stabilised with `+ 1e-10`.

### Evaluation

Genuine scores (same subject) and impostor scores (different subjects) are generated from an 80/20 per-subject enroll/probe split. ROC curves drive EER, AUC, d-prime, TMR/FMR and Rank-1 metrics, all visualised in a 2×3 matplotlib dashboard.

---

## Results

Indicative performance on the ORL dataset (actual values vary per run):

| Metric | PCA (Eigenfaces) | LBP |
|---|---|---|
| EER | ~5–12% | ~3–8% |
| AUC | ~0.95–0.99 | ~0.97–0.99 |
| d-prime | ~2.5–4.0 | ~3.0–5.0 |
| Rank-1 Accuracy | ~85–95% | ~90–97% |
| TMR @ 1% FMR | ~85–98% | ~90–99% |

LBP generally outperforms PCA on ORL due to its illumination tolerance; PCA benefits significantly from histogram equalisation.

---

## Configuration

Key parameters live in the `Config` class in `enrollment.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `IMG_SIZE` | `(100, 100)` | Normalised face crop size |
| `LBP_GRID` | `5` | Spatial grid for LBP histograms |
| `LBP_UNIFORM` | `True` | Uniform LBP (59 bins) vs full 256 |
| `CLAHE_CLIP` | `2.5` | Local contrast clipping limit |
| `MAX_YAW / MAX_PITCH / MAX_ROLL` | `30 / 25 / 30` | Head-pose rejection thresholds (deg) |
| `LIVENESS_ENABLED` | `False` | Blink-based liveness check |
| `DEFAULT_VERIFY_THRESHOLD` | `0.80` | Fallback match threshold |
| `DEFAULT_DUPLICATE_THRESHOLD` | `0.82` | Fallback duplicate-enroll threshold |

Thresholds in `calibration.json` override the defaults when present.

---

## Development

There is no formal test suite in the repository. To exercise the system manually:

1. Run `python main.py` and confirm the dashboard PNG is generated.
2. Run `python app.py` and use the browser UI to enroll and verify a subject with a webcam.

When adding features, keep the NumPy-vectorised style used in `main.py` and `enrollment.py`, and avoid introducing new heavy dependencies without need.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the existing code style (no external formatter is configured).
3. Ensure `python main.py` and `python app.py` still run cleanly.
4. Open a pull request describing the change and any metric impact.

---

## References

1. Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition.* Journal of Cognitive Neuroscience, 3(1), 71–86.
2. Ojala, T., Pietikäinen, M. & Mäenpää, T. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns.* IEEE TPAMI, 24(7), 971–987.
3. Samaria, F. & Harter, A. (1994). *Parameterisation of a Stochastic Model for Human Face Identification.* IEEE Workshop on Applications of Computer Vision.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
5. Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
6. ISO/IEC 19795-1:2021 — *Biometric Performance Testing and Reporting.*

See [`REPORT.md`](REPORT.md) for the full technical report.

---

## License

No license file is included in this repository. Contact the repository owner for licensing terms before reuse.
