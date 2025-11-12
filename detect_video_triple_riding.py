from __future__ import annotations
import argparse
import csv
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import subprocess
import concurrent.futures
import multiprocessing

def process_frame_with_model(frame_path, model_path, conf, device, imgsz, max_det, iou_thresh, dist_thresh,
                           out_folder, compact_violation, show_confidence, show_arrow, violation_only_box,
                           quiet, db_manager=None):
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image, ImageDraw

    frame_number = int(frame_path.stem.split('_')[1])

    model = YOLO(str(model_path))

    results = model.predict(source=str(frame_path), conf=conf, device=device,
                            imgsz=imgsz, max_det=max_det, save=False, verbose=False)
    r = results[0]

    persons = []
    vehicles = []

    try:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy().item()) if hasattr(box.cls, 'cpu') else int(box.cls)
            xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy[0]

            if cls == 0:  # person
                persons.append({
                    "bbox": xyxy,
                    "conf": float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                })
            elif cls == 3:  # motorbike
                vehicles.append({
                    "bbox": xyxy,
                    "conf": float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                })
    except Exception as e:
        print(f"Warning: Error processing boxes for {frame_path.name}: {e}")
        return None

    vehicle_groups = group_riders_with_vehicles(
        persons, vehicles,
        iou_threshold=iou_thresh,
        distance_threshold=dist_thresh
    )

    violations = sum(1 for g in vehicle_groups if g["is_violation"])

    # Save annotated frame if there are any detections (vehicles or riders)
    if len(vehicle_groups) > 0 or len(persons) > 0:
        if violations > 0:
            out_path = Path(out_folder) / f"violation_frame_{frame_number:06d}.jpg"
        else:
            out_path = Path(out_folder) / f"frame_{frame_number:06d}.jpg"
        annotate_image(str(frame_path), vehicle_groups, out_path,
                       compact_violation=compact_violation,
                       show_confidence=show_confidence,
                       show_arrow=show_arrow,
                       violation_only_box=violation_only_box)

    if not quiet:
        print_detection_report(frame_path.name, vehicle_groups)

    if db_manager:
        result_doc = {
            "name": f"frame_{frame_number:06d}",
            "type": "video_frame",
            "frame_number": frame_number,
            "total_vehicles": len(vehicles),
            "total_riders": len(persons),
            "violations": violations,
            "status": "VIOLATION" if violations > 0 else "COMPLIANT",
            "timestamp": datetime.now().isoformat(),
            "processed_at": datetime.now().isoformat(),
            "frame_path": str(frame_path),
            "video_source": getattr(args, 'file', 'unknown') if 'args' in locals() else 'unknown'
        }
        try:
            db_manager.insert_result(result_doc)
            if not quiet:
                print(f"✓ Stored frame {frame_number} result in database")
        except Exception as e:
            print(f"Warning: Failed to insert result for frame {frame_number} into database: {e}")

    return (frame_number, len(vehicles), len(persons), violations)

def download_kaggle_dataset(dataset_slug: str, file_name: str, temp_dir: Path) -> Path:
    """Download a specific file from Kaggle dataset to temp directory."""
    try:
        # Use curl to download the specific file
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}/{file_name}"
        video_path = temp_dir / file_name
        cmd = ["curl", "-L", "-o", str(video_path), url]
        subprocess.run(cmd, check=True)

        if not video_path.exists():
            raise FileNotFoundError(f"File {file_name} not found after download")

        return video_path
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to download dataset: {e}") from e

def extract_frames(video_path: Path, temp_frames_dir: Path, frame_interval: int = 30) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = temp_frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)

        frame_count += 1

    cap.release()
    return frames

def find_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection + 1e-6)

def group_riders_with_vehicles(persons: List[Dict], vehicles: List[Dict],
                             iou_threshold: float = 0.05,
                             distance_threshold: float = 0.75) -> List[Dict]:
    vehicle_groups = []
    used_persons = set()  

    for vehicle in vehicles:
        v_box = np.array(vehicle["bbox"]) 
        v_center = np.array([(v_box[0] + v_box[2])/2, (v_box[1] + v_box[3])/2])
        v_width = v_box[2] - v_box[0]
        v_height = v_box[3] - v_box[1]

        riders = []
        candidates = [] 

        for i, person in enumerate(persons):
            if i in used_persons:
                continue

            p_box = np.array(person["bbox"])
            p_center = np.array([(p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2])

            dist = np.linalg.norm(v_center - p_center) / max(v_width, v_height)

            horiz_dist = abs(p_center[0] - v_center[0]) / max(v_width, 1.0)

            iou = compute_iou(v_box, p_box)

            p_bottom = p_box[3]
            v_top = v_box[1]
            v_bottom = v_box[3]

            top_margin = 0.05 * v_height   
            bottom_margin = 0.30 * v_height

            bottom_overlap = (p_bottom >= (v_top - top_margin)) and (p_bottom <= (v_bottom + bottom_margin))

            if iou > iou_threshold or (horiz_dist < 1.2 and bottom_overlap):
                candidates.append((person, dist))

        candidates.sort(key=lambda x: x[1])  
        max_riders = 4  
        for person, _ in candidates[:max_riders]:
            for idx, p in enumerate(persons):
                if np.array_equal(p["bbox"], person["bbox"]):
                    if idx not in used_persons:
                        riders.append(person)
                        used_persons.add(idx)
                    break

        is_violation = False
        if len(riders) >= 3:
            rider_centers = []
            for r in riders:
                r_box = np.array(r["bbox"])
                r_center = np.array([(r_box[0] + r_box[2])/2, (r_box[1] + r_box[3])/2])
                rider_centers.append(r_center)

            if len(rider_centers) > 1:
                centers = np.array(rider_centers)
                spread = np.max(centers, axis=0) - np.min(centers, axis=0)
                min_spread = v_width * 0.3  
                is_violation = spread[0] > min_spread or spread[1] > min_spread

        vehicle_groups.append({
            "vehicle": vehicle,
            "riders": riders,
            "is_violation": is_violation,
            "rider_count": len(riders)
        })

    return vehicle_groups

def annotate_image(image_path: str, vehicle_groups: List[Dict],
                  out_path: Path,
                  compact_violation: bool = False,
                  show_confidence: bool = False,
                  show_arrow: bool = False,
                  violation_only_box: bool = False) -> None:
    """Draw violation markers and counts on the image."""
    try:
        img = None
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception:
            try:
                import cv2
                import numpy as np
                cv_img = cv2.imread(str(image_path))
                if cv_img is not None:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv_img)
            except Exception:
                print(f"Warning: Could not open image {image_path}")
                return

        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return

        draw = ImageDraw.Draw(img)

        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except Exception:
            title_font = label_font = ImageFont.load_default()


    except Exception as e:
        print(f"Warning: Could not open image {image_path}: {e}")
        return


    if violation_only_box:
        viol_groups = [g for g in vehicle_groups if g.get("is_violation")]
        if viol_groups:
            xs = []
            ys = []
            confidences = []
            for g in viol_groups:
                v = g["vehicle"]
                x1, y1, x2, y2 = map(float, v["bbox"])
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                if v.get("conf") is not None:
                    confidences.append(float(v.get("conf")))
                for r in g.get("riders", []):
                    rx1, ry1, rx2, ry2 = map(float, r["bbox"])
                    xs.extend([rx1, rx2])
                    ys.extend([ry1, ry2])

            bx1, bx2 = min(xs), max(xs)
            by1, by2 = min(ys), max(ys)

            for w in range(4):
                draw.rectangle([bx1-w, by1-w, bx2+w, by2+w], outline=(255, 0, 0))

            label = "TRIPLE RIDING VIOLATION!"
            tw = draw.textlength(label, font=label_font)
            draw.rectangle([bx1, by1-30, bx1+tw+10, by1], fill=(255,0,0))
            draw.text((bx1+5, by1-28), label, fill='white', font=label_font)

            if show_confidence and confidences:
                conf_text = f"avg_conf: {float(np.mean(confidences)):.2f}"
                draw.text((bx1+5, by1-55), conf_text, fill='white', font=label_font)

            if show_arrow:
                ax = (bx1 + bx2) / 2
                ay = by1 - 60
                draw.polygon([(ax, ay), (ax-8, ay+16), (ax+8, ay+16)], fill=(255,0,0))

            img.save(out_path)
            return
        
    for group in vehicle_groups:
        vehicle = group["vehicle"]
        riders = group["riders"]
        rider_count = group.get("rider_count", len(riders))
        is_violation = group["is_violation"]

        x1, y1, x2, y2 = map(float, vehicle["bbox"])

        violation_color = (255, 0, 0)  # Red
        normal_color = (255, 255, 0)   # Yellow
        box_color = violation_color if is_violation else normal_color

        if compact_violation and is_violation:
            xs = [x1, x2]
            ys = [y1, y2]
            for r in riders:
                rx1, ry1, rx2, ry2 = map(float, r["bbox"])
                xs.extend([rx1, rx2])
                ys.extend([ry1, ry2])
            bx1, bx2 = min(xs), max(xs)
            by1, by2 = min(ys), max(ys)

            for w in range(4):
                draw.rectangle([bx1-w, by1-w, bx2+w, by2+w], outline=violation_color)

            label = "TRIPLE RIDING VIOLATION!"
            tw = draw.textlength(label, font=label_font)
            draw.rectangle([bx1, by1-30, bx1+tw+10, by1], fill=violation_color)
            draw.text((bx1+5, by1-28), label, fill='white', font=label_font)

            if show_confidence and vehicle.get("conf") is not None:
                conf_text = f"conf: {vehicle['conf']:.2f}"
                draw.text((bx1+5, by1-55), conf_text, fill='white', font=label_font)

            if show_arrow:
                ax = (bx1 + bx2) / 2
                ay = by1 - 60
                draw.polygon([(ax, ay), (ax-8, ay+16), (ax+8, ay+16)], fill=violation_color)

        else:
            for w in range(3):  
                draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=box_color)

            if show_confidence and vehicle.get("conf") is not None:
                conf_text = f"{vehicle['conf']:.2f}"
                draw.text((x1+4, y1-20), conf_text, fill=box_color, font=label_font)
            if show_arrow:
                ax = (x1 + x2) / 2
                ay = y1 - 30
                draw.polygon([(ax, ay), (ax-6, ay+12), (ax+6, ay+12)], fill=box_color)
            for i, rider in enumerate(riders, 1):
                rx1, ry1, rx2, ry2 = map(float, rider["bbox"])
                for w in range(2):
                    draw.rectangle([rx1-w, ry1-w, rx2+w, ry2+w], outline=box_color)
                text = f"Rider {i}"
                tw = draw.textlength(text, font=label_font)
                text_bg = [rx1, ry1-30, rx1+tw+8, ry1-5]
                draw.rectangle(text_bg, fill=box_color)
                draw.text((rx1+4, ry1-28), text, fill='black', font=label_font)
                v_center = [(x1 + x2)/2, (y1 + y2)/2]
                r_center = [(rx1 + rx2)/2, (ry1 + ry2)/2]
                for w in range(2):
                    draw.line([v_center[0], v_center[1], r_center[0], r_center[1]],
                             fill=box_color, width=2-w)

    img.save(out_path)

def _color_text(text: str, color: str) -> str:
    colors = {
        'reset': '\033[0m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'cyan': '\033[36m',
    }
    if color not in colors:
        return text
    return f"{colors[color]}{text}{colors['reset']}"

def print_detection_report(frame_name: str, vehicle_groups: List[Dict], color: bool = True):
    sep = "=" * 50
    print("\n" + sep)
    if color:
        print(_color_text(f"Detection Report for: {frame_name}", 'cyan'))
    else:
        print(f"Detection Report for: {frame_name}")
    print(sep)

    violation_count = sum(1 for g in vehicle_groups if g["is_violation"])
    total_vehicles = len(vehicle_groups)
    total_riders = sum(len(g["riders"]) for g in vehicle_groups)

    print(f"\nSummary:")
    print(f"- Total Vehicles Detected: {total_vehicles}")
    print(f"- Total Riders Detected: {total_riders}")
    if color and violation_count > 0:
        print(_color_text(f"- Triple Riding Violations: {violation_count}", 'red'))
    else:
        print(f"- Triple Riding Violations: {violation_count}")

    if violation_count > 0:
        print("\nViolation Details:")
        for i, group in enumerate(vehicle_groups, 1):
            if group["is_violation"]:
                title = f"Vehicle {i}:"
                if color:
                    print(_color_text(f"\n{title}", 'red'))
                    print(_color_text(f"- Number of Riders: {len(group['riders'])}", 'yellow'))
                    print(_color_text("- Status: ⚠️ TRIPLE RIDING VIOLATION", 'red'))
                else:
                    print(f"\n{title}")
                    print(f"- Number of Riders: {len(group['riders'])}")
                    print("- Status: ⚠️ TRIPLE RIDING VIOLATION")
    else:
        # When no violations, print a compliant note
        if color:
            print(_color_text("\nNo triple-riding violations detected in this frame.", 'green'))
        else:
            print("\nNo triple-riding violations detected in this frame.")

    print(f"\n{sep}")

def main(args=None):
    if args is None:
        p = argparse.ArgumentParser()
        p.add_argument("--video", "-v", help="Kaggle dataset slug (e.g., arunavfc11/indian-traffic-videos)")
        p.add_argument("--file", "-f", default="13143934_3840_2160_30fps.mp4", help="Video file name in the dataset")
        p.add_argument("--weights", "-w", default="yolov8n.pt", help="Path to YOLO weights")
        p.add_argument("--out", "-o", default="output video", help="Annotated output folder")
        p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
        p.add_argument("--imgsz", type=int, default=640, help="Inference image size (pixels)")
        p.add_argument("--max-det", type=int, default=300, help="Maximum number of detections per image")
        p.add_argument("--iou-thresh", type=float, default=0.1, help="IoU threshold for grouping riders with vehicles")
        p.add_argument("--dist-thresh", type=float, default=0.75, help="Distance threshold for grouping")
        p.add_argument("--frame-interval", type=int, default=30, help="Process every Nth frame")
        p.add_argument("--compact-violation", action="store_true", help="Draw single compact box for violations")
        p.add_argument("--show-confidence", action="store_true", help="Show detection confidence")
        p.add_argument("--show-arrow", action="store_true", help="Draw direction arrow")
        p.add_argument("--device", default=None, help="Device for inference")
        p.add_argument("--quiet", action="store_true", help="Suppress verbose output")
        p.add_argument("--no-color", action="store_true", help="Disable colored terminal output")
        p.add_argument("--violation-only-box", action="store_true", help="Draw one combined box for all violations")
        p.add_argument("--mongodb-uri", default="mongodb://localhost:27017/", help="MongoDB connection URI")
        p.add_argument("--db-name", default="triple_riding_db", help="MongoDB database name")
        p.add_argument("--collection-name", default="video_detections", help="MongoDB collection name for video results")
        args = p.parse_args()

    try:
        from ultralytics import YOLO
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as e:
        raise SystemExit("Please install required packages: pip install ultralytics pillow numpy opencv-python kaggle") from e
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if args.video:
            print(f"Downloading video from Kaggle dataset: {args.video}")
            video_path = download_kaggle_dataset(args.video, args.file, temp_path)
        else:
            local_video_path = Path(args.file)
            if not local_video_path.exists():
                raise SystemExit(f"Local video file not found: {local_video_path}")
            video_path = temp_path / local_video_path.name
            shutil.copy2(local_video_path, video_path)
            print(f"Using local video file: {local_video_path}")

        temp_frames_dir = temp_path / "frames"
        temp_frames_dir.mkdir()

        print(f"Extracting frames every {args.frame_interval} frames...")
        frame_paths = extract_frames(video_path, temp_frames_dir, args.frame_interval)

        if not frame_paths:
            raise SystemExit("No frames extracted from video")
        video_path.unlink()
        out_folder = Path(args.out)
        out_folder.mkdir(parents=True, exist_ok=True)

        model_path = Path(args.weights)
        if not model_path.exists():
            raise SystemExit(f"Weights not found: {model_path}")

        model = YOLO(str(model_path))
        from database import create_database_manager
        db_manager = None
        try:
            db_manager = create_database_manager(args.mongodb_uri, args.db_name, args.collection_name)
            print(f"✓ Connected to MongoDB: {args.db_name}.{args.collection_name}")
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB: {e}. Proceeding without database storage.")
        report_rows: List[Tuple[int, int, int, int]] = []

        def process_frame(frame_path):
            frame_number = int(frame_path.stem.split('_')[1])
            results = model.predict(source=str(frame_path), conf=args.conf, device=args.device,
                                    imgsz=args.imgsz, max_det=args.max_det, save=False, verbose=False)
            r = results[0]
            persons = []
            vehicles = []

            try:
                for box in r.boxes:
                    cls = int(box.cls.cpu().numpy().item()) if hasattr(box.cls, 'cpu') else int(box.cls)
                    xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy[0]

                    if cls == 0: 
                        persons.append({
                            "bbox": xyxy,
                            "conf": float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        })
                    elif cls == 3: 
                        vehicles.append({
                            "bbox": xyxy,
                            "conf": float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        })
            except Exception as e:
                print(f"Warning: Error processing boxes for {frame_path.name}: {e}")
                return None

            vehicle_groups = group_riders_with_vehicles(
                persons, vehicles,
                iou_threshold=args.iou_thresh,
                distance_threshold=args.dist_thresh
            )

            violations = sum(1 for g in vehicle_groups if g["is_violation"])

            # Save annotated frame if there are any detections (vehicles or riders)
            if len(vehicle_groups) > 0 or len(persons) > 0:
                if violations > 0:
                    out_path = out_folder / f"violation_frame_{frame_number:06d}.jpg"
                else:
                    out_path = out_folder / f"frame_{frame_number:06d}.jpg"
                annotate_image(str(frame_path), vehicle_groups, out_path,
                               compact_violation=args.compact_violation,
                               show_confidence=args.show_confidence,
                               show_arrow=args.show_arrow,
                               violation_only_box=args.violation_only_box)
                if not args.quiet:
                    status = "VIOLATION" if violations > 0 else "COMPLIANT"
                    print(f"  Saved frame {frame_number}: {len(vehicle_groups)} vehicles, {len(persons)} riders - {status}")

            if not args.quiet:
                print_detection_report(frame_path.name, vehicle_groups)

            return (frame_number, len(vehicles), len(persons), violations)

        num_workers = min(multiprocessing.cpu_count(), len(frame_paths))
        for frame_path in frame_paths:
            result = process_frame_with_model(frame_path, model_path,
                                             args.conf, args.device, args.imgsz, args.max_det,
                                             args.iou_thresh, args.dist_thresh, args.out,
                                             args.compact_violation, args.show_confidence, args.show_arrow,
                                             args.violation_only_box, args.quiet, db_manager)
            if result:
                report_rows.append(result)

        # Write CSV report
        report_file = out_folder / "report.csv"
        with report_file.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["Frame Number","Total Vehicles","Total Riders","Triple Riding Violations"])
            for row in report_rows:
                writer.writerow(row)
        total_frames = len(report_rows)
        total_vehicles = sum(row[1] for row in report_rows)
        total_riders = sum(row[2] for row in report_rows)
        total_violations = sum(row[3] for row in report_rows)

        print("\n" + "="*50)
        print("TRIPLE RIDING DETECTION SUMMARY")
        print("="*50)
        print(f"Total Frames Processed: {total_frames}")
        print(f"Total Vehicles Detected: {total_vehicles}")
        print(f"Total Riders Detected: {total_riders}")
        print(f"Triple Riding Violations: {total_violations}")
        print("="*50)

        if sys.stdout.isatty():
            print(' ' * 80, end='\r', flush=True)
        print(f"Done. Report written to: {report_file}")
        if db_manager:
            print(f"Results also stored in MongoDB: {args.db_name}.{args.collection_name}")
        if db_manager:
            db_manager.close()

if __name__ == "__main__":
    dataset_slug = "arunavfc11/indian-traffic-videos"
    file_name = "13105476_3840_2160_30fps.mp4"
    quiet = True  

    class Args:
        def __init__(self):
            self.video = dataset_slug
            self.file = file_name
            self.weights = "yolov8n.pt"
            self.out = "output video"
            self.conf = 0.2  
            self.imgsz = 640
            self.max_det = 300
            self.iou_thresh = 0.05  
            self.dist_thresh = 0.5  
            self.frame_interval = 60  
            self.compact_violation = False
            self.show_confidence = False
            self.show_arrow = False
            self.device = None
            self.quiet = quiet
            self.no_color = False
            self.violation_only_box = False
            self.mongodb_uri = "mongodb://localhost:27017/"
            self.db_name = "triple_riding_db"
            self.collection_name = "video_detections"

    args = Args()
    main(args)

    print("\nProcessing local video file with optimized detection settings...")
    class TestArgs:
        def __init__(self):
            self.video = None  # No dataset for local file
            self.file = "datasets/triple_riding/video/test.mp4"
            self.weights = "yolov8n.pt"
            self.out = "outputs/triple_riding/video"
            # OPTIMIZED SETTINGS FOR LOCAL VIDEO (matching image detection)
            self.conf = 0.3  # Better confidence threshold for cleaner detections
            self.imgsz = 640
            self.max_det = 300
            self.iou_thresh = 0.1  # Better IoU threshold for accurate grouping
            self.dist_thresh = 0.75  # Better distance threshold for rider association
            self.frame_interval = 30  # Process more frames for better coverage
            self.compact_violation = True  # Draw compact violation boxes
            self.show_confidence = True  # Show confidence scores
            self.show_arrow = True  # Show direction arrows
            self.device = None
            self.quiet = quiet
            self.no_color = False
            self.violation_only_box = False
            # KEEP MONGODB FOR TIMESTAMP REPORTING
            self.mongodb_uri = "mongodb://localhost:27017/"
            self.db_name = "triple_riding_db"
            self.collection_name = "video_detections"

    test_args = TestArgs()
    main(test_args)
