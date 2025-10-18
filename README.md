# Triple Riding Detection

This repository provides a pipeline to detect and report instances of triple riding (people on a motorcycle) using a YOLOv8 model and a small detection script. It includes a convenience wrapper to set up a virtual environment, install dependencies, and run detection with recommended defaults.

Contents
--------

- `detect_triple_riding.py` — main detection and reporting script.
- `yolov8n.pt` — example model weights (small YOLOv8 model).
- `datasets/triple_riding/` — images and optional YOLO labels.
- `scripts/run_all.sh` — helper script to automate setup + run.

Quick start (recommended)
-------------------------

Run everything in one step. The wrapper will create `.venv` (if missing), install dependencies, and run the detector:

```bash
./scripts/run_all.sh
```

If you already installed dependencies and want to skip installation:

```bash
./scripts/run_all.sh --no-install
```

Forward extra arguments to `detect_triple_riding.py` by using `--`. Example (change confidence):

```bash
./scripts/run_all.sh -- --conf 0.5
```

Manual setup and run
--------------------

1) Create and activate a virtualenv (zsh/bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics pillow numpy opencv-python
```

3) Run the detector with the recommended flags:

```bash
python3 detect_triple_riding.py \
  --images datasets/triple_riding/images \
  --weights yolov8n.pt \
  --conf 0.3 --dist-thresh 0.75 \
  --compact-violation --show-confidence --show-arrow
```

Or as a single one-liner (create venv, install, run):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install ultralytics pillow numpy opencv-python && python3 detect_triple_riding.py --images datasets/triple_riding/images --weights yolov8n.pt --conf 0.3 --dist-thresh 0.75 --compact-violation --show-confidence --show-arrow
```

Data layout
-----------

Place images here:

```
datasets/triple_riding/images/train/
datasets/triple_riding/images/val/
```

If you have YOLO-format labels, add them at:

```
datasets/triple_riding/labels/train/
datasets/triple_riding/labels/val/
```

Outputs
-------

- Annotated images are written to `outputs/triple_riding/images/` (or `runs/detect/` depending on the YOLO runtime).
- A CSV summary report (example: `outputs/triple_riding/report.csv` or `triple_report.csv`) contains per-image counts and violation details.

Useful commands & tips
----------------------

- Show detector help:

```bash
python3 detect_triple_riding.py --help
```

- Force CPU (no GPU):

```bash
python3 detect_triple_riding.py --device cpu ...
```

- Deactivate virtualenv when done:

```bash
deactivate
```

Notes for maintainers
--------------------

- The script uses COCO class indices by default (person=0, motorbike=3). If you use a custom-trained model with different class indexing, update `detect_triple_riding.py` accordingly.
- `scripts/run_all.sh` executes the Python binary inside `.venv` directly; activating the venv is optional for interactive use but not required by the wrapper.

Further improvements (optional)
-----------------------------

- Add CLI flags to `scripts/run_all.sh` to override `--images`, `--weights`, etc., without using `--` for forwarding.
- Add example outputs (screenshots) or a short CONTRIBUTING guide.

License & contact
-----------------

This project is provided as-is. Open an issue for bugs, feature requests, or questions.

---

If you'd like, I can implement one of the optional improvements above — tell me which and I will add it.
Triple Riding dataset
---------------------

Place your images (downloaded manually from the web) in:

  datasets/triple_riding/images/train/
  datasets/triple_riding/images/val/

Make sure filenames are unique and image formats are supported (jpg, png, webp, bmp).

Labels (optional at detection time):
- If you already have YOLO labels, put them in `datasets/triple_riding/labels/train` and `.../val` with the same basename as images.

Run detection (does not change existing models):

  python detect_triple_riding.py --images datasets/triple_riding/images --weights yolov8n.pt --out outputs/triple_riding/images

Outputs
-------
- Annotated images are saved into `outputs/triple_riding/images/`.
- A `report.csv` summarizing person and motorbike counts is saved to `outputs/triple_riding/report.csv`.

Notes
-----
- The default script uses class indices from COCO (person=0, motorbike=3). If you fine-tune a custom model with different classes, update the script's class mapping.
- This workflow keeps your triple-riding detection separate from the existing helmet model and weights.