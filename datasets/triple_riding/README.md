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