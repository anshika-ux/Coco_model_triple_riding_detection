Triple-riding detection — how to run

This file contains the instructions to run the triple-riding detector (same
content I can append into `README.md` if you want).

1) Install (recommended inside a venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics pillow numpy opencv-python
```

2) Prepare images

Place your images (downloaded manually from the web) under:

- datasets/triple_riding/images/train/
- datasets/triple_riding/images/val/

The script will search recursively under the folder you pass with `--images`.

3) Run detector examples

- Compact (single violation box per vehicle + show confidence + arrow):

```bash
python3 detect_triple_riding.py --images datasets/triple_riding/images \
  --weights yolov8n.pt --conf 0.3 --dist-thresh 0.75 \
  --compact-violation --show-confidence --show-arrow
```

4) Output

- Annotated images: outputs/triple_riding/images/
- A CSV-style summary is printed to the terminal when the run finishes.

5) Tips

- To reduce false positives, try a larger model (yolov8s.pt), increase
  `--conf`, or change `--dist-thresh` (how far a person can be from a vehicle
  to still be considered its rider).
- If you want the CSV saved to disk instead of printed, tell me and I will
  re-enable file output.

Append this content into `README.md` manually with:

```bash
cat docs/triple_riding.md >> README.md
```

(or ask me to append it for you and I'll try again).