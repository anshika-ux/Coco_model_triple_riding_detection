
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute intersection over union between two boxes [x1,y1,x2,y2]."""
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
    """Group person detections with their likely vehicle using IoU and distance."""
    vehicle_groups = []
    used_persons = set()  # Track assigned riders
    
    for vehicle in vehicles:
        v_box = np.array(vehicle["bbox"])  # [x1,y1,x2,y2]
        v_center = np.array([(v_box[0] + v_box[2])/2, (v_box[1] + v_box[3])/2])
        v_width = v_box[2] - v_box[0]
        v_height = v_box[3] - v_box[1]
        
        # Find riders for this vehicle
        riders = []
        candidates = []  # Store (person, distance) pairs
        
        for i, person in enumerate(persons):
            if i in used_persons:
                continue
                
            p_box = np.array(person["bbox"])
            p_center = np.array([(p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2])
            
            # Normalized distance relative to vehicle size
            dist = np.linalg.norm(v_center - p_center) / max(v_width, v_height)

            # Horizontal normalized distance
            horiz_dist = abs(p_center[0] - v_center[0]) / max(v_width, 1.0)

            # Compute IoU
            iou = compute_iou(v_box, p_box)

            # Require either a reasonable IoU OR that the person's bottom edge overlaps the vehicle's vertical range
            p_bottom = p_box[3]
            v_top = v_box[1]
            v_bottom = v_box[3]

            # margin allowances (relative to vehicle height)
            top_margin = 0.05 * v_height    # allow a bit above vehicle top
            bottom_margin = 0.30 * v_height # allow a bit below vehicle bottom (sitting legs)

            bottom_overlap = (p_bottom >= (v_top - top_margin)) and (p_bottom <= (v_bottom + bottom_margin))

            # Accept candidate if IoU strong or horizontal alignment plus bottom overlap
            if iou > iou_threshold or (horiz_dist < 1.2 and bottom_overlap):
                candidates.append((person, dist))
        
        # Sort by distance and take the closest N riders
        candidates.sort(key=lambda x: x[1])  # Sort by distance
        max_riders = 4  # Maximum reasonable number of riders to consider per vehicle
        
        for person, _ in candidates[:max_riders]:
            # Find matching person by comparing bbox
            for idx, p in enumerate(persons):
                if np.array_equal(p["bbox"], person["bbox"]):
                    if idx not in used_persons:
                        riders.append(person)
                        used_persons.add(idx)
                    break
        
        vehicle_groups.append({
            "vehicle": vehicle,
            "riders": riders,
            "is_violation": len(riders) >= 3,
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
        # Try multiple methods to open the image
        img = None
        try:
            # Try standard PIL open
            img = Image.open(image_path).convert('RGB')
        except Exception:
            try:
                # Try reading with OpenCV and convert to PIL
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
        
        # Try to load fonts with different sizes
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except Exception:
            title_font = label_font = ImageFont.load_default()
            
    # (Header removed) — images will not include the black headline/title
            
    except Exception as e:
        print(f"Warning: Could not open image {image_path}: {e}")
        return
    
    # If the caller requested a single combined box for all violations, and there are violations,
    # compute a single bounding box that covers all vehicles and riders involved in violations
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

            # Draw bold combined box
            for w in range(4):
                draw.rectangle([bx1-w, by1-w, bx2+w, by2+w], outline=(255, 0, 0))

            # Label
            label = "TRIPLE RIDING VIOLATION!"
            tw = draw.textlength(label, font=label_font)
            draw.rectangle([bx1, by1-30, bx1+tw+10, by1], fill=(255,0,0))
            draw.text((bx1+5, by1-28), label, fill='white', font=label_font)

            # Optionally show average confidence
            if show_confidence and confidences:
                conf_text = f"avg_conf: {float(np.mean(confidences)):.2f}"
                draw.text((bx1+5, by1-55), conf_text, fill='white', font=label_font)

            # Optionally show arrow
            if show_arrow:
                ax = (bx1 + bx2) / 2
                ay = by1 - 60
                draw.polygon([(ax, ay), (ax-8, ay+16), (ax+8, ay+16)], fill=(255,0,0))

            img.save(out_path)
            return

    # Default behavior: draw per-vehicle and rider boxes as before
    for group in vehicle_groups:
        vehicle = group["vehicle"]
        riders = group["riders"]
        rider_count = group.get("rider_count", len(riders))
        is_violation = group["is_violation"]

        # Get coordinates and status
        x1, y1, x2, y2 = map(float, vehicle["bbox"])

        # Colors for different elements
        violation_color = (255, 0, 0)  # Red
        normal_color = (255, 255, 0)   # Yellow
        box_color = violation_color if is_violation else normal_color

        # If compact_violation is requested and this is a violation, draw a single box
        if compact_violation and is_violation:
            # Compute combined bbox covering vehicle and riders
            xs = [x1, x2]
            ys = [y1, y2]
            for r in riders:
                rx1, ry1, rx2, ry2 = map(float, r["bbox"])
                xs.extend([rx1, rx2])
                ys.extend([ry1, ry2])
            bx1, bx2 = min(xs), max(xs)
            by1, by2 = min(ys), max(ys)

            # Draw a single bold box for the violation
            for w in range(4):
                draw.rectangle([bx1-w, by1-w, bx2+w, by2+w], outline=violation_color)

            # Draw label
            label = "TRIPLE RIDING VIOLATION!"
            tw = draw.textlength(label, font=label_font)
            draw.rectangle([bx1, by1-30, bx1+tw+10, by1], fill=violation_color)
            draw.text((bx1+5, by1-28), label, fill='white', font=label_font)

            # Optionally show confidence of the vehicle (use vehicle.conf)
            if show_confidence and vehicle.get("conf") is not None:
                conf_text = f"conf: {vehicle['conf']:.2f}"
                draw.text((bx1+5, by1-55), conf_text, fill='white', font=label_font)

            # Optionally draw a simple up-arrow above the box to indicate direction
            if show_arrow:
                ax = (bx1 + bx2) / 2
                ay = by1 - 60
                draw.polygon([(ax, ay), (ax-8, ay+16), (ax+8, ay+16)], fill=violation_color)

        else:
            # Draw vehicle box with clean lines
            for w in range(3):  # Draw multiple lines for thickness
                draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=box_color)

            # Optionally show confidence on vehicle
            if show_confidence and vehicle.get("conf") is not None:
                conf_text = f"{vehicle['conf']:.2f}"
                draw.text((x1+4, y1-20), conf_text, fill=box_color, font=label_font)

            # Optionally draw arrow for direction
            if show_arrow:
                ax = (x1 + x2) / 2
                ay = y1 - 30
                draw.polygon([(ax, ay), (ax-6, ay+12), (ax+6, ay+12)], fill=box_color)

            # Draw rider boxes with numbers
            for i, rider in enumerate(riders, 1):
                rx1, ry1, rx2, ry2 = map(float, rider["bbox"])

                # Clean rider box
                for w in range(2):
                    draw.rectangle([rx1-w, ry1-w, rx2+w, ry2+w], outline=box_color)

                # Add rider number with background
                text = f"Rider {i}"
                tw = draw.textlength(text, font=label_font)
                text_bg = [rx1, ry1-30, rx1+tw+8, ry1-5]
                draw.rectangle(text_bg, fill=box_color)
                draw.text((rx1+4, ry1-28), text, fill='black', font=label_font)

                # Draw connection line with multiple passes for visibility
                v_center = [(x1 + x2)/2, (y1 + y2)/2]
                r_center = [(rx1 + rx2)/2, (ry1 + ry2)/2]
                for w in range(2):
                    draw.line([v_center[0], v_center[1], r_center[0], r_center[1]], 
                             fill=box_color, width=2-w)
    
    img.save(out_path)


def _color_text(text: str, color: str) -> str:
    """Return text wrapped with ANSI color codes. color in {'red','green','yellow','cyan','reset'}"""
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


def print_detection_report(image_name: str, vehicle_groups: List[Dict], color: bool = True):
    """Print a formatted detection report for an image with optional colors."""
    sep = "=" * 50
    print("\n" + sep)
    if color:
        print(_color_text(f"Detection Report for: {image_name}", 'cyan'))
    else:
        print(f"Detection Report for: {image_name}")
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
            print(_color_text("\nNo triple-riding violations detected in this image.", 'green'))
        else:
            print("\nNo triple-riding violations detected in this image.")

    print(f"\n{sep}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", "-i", default="datasets/triple_riding/images", 
                  help="Folder with images (will search recursively)")
    p.add_argument("--weights", "-w", default="yolov8n.pt", 
                  help="Path to YOLO weights (default repo yolov8n.pt)")
    p.add_argument("--out", "-o", default="outputs/triple_riding/images", 
                  help="Annotated output folder")
    p.add_argument("--conf", type=float, default=0.35, 
                  help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                  help="Inference image size (pixels). Increase for higher accuracy, decrease for speed")
    p.add_argument("--max-det", type=int, default=300,
                  help="Maximum number of detections per image (higher may increase recall)")
    p.add_argument("--iou-thresh", type=float, default=0.1,
                  help="IoU threshold for grouping riders with vehicles")
    p.add_argument("--dist-thresh", type=float, default=0.75,
                  help="Distance threshold for grouping (relative to vehicle size)")
    p.add_argument("--compact-violation", action="store_true",
                  help="If set, draw a single compact box for violations instead of rider boxes/numbers")
    p.add_argument("--show-confidence", action="store_true",
                  help="If set, show detection confidence on vehicle boxes")
    p.add_argument("--show-arrow", action="store_true",
                  help="If set, draw a small direction arrow above each vehicle")
    p.add_argument("--device", default=None, 
                  help="Device for inference, e.g. 'cpu' or '0' for GPU")
    p.add_argument("--quiet", action="store_true",
                  help="Suppress verbose per-image output and ultralytics progress")
    p.add_argument("--no-color", action="store_true",
                  help="Disable colored terminal output")
    p.add_argument("--show-summary", action="store_true",
                  help="Print the final Detection Summary and visualization guide (hidden by default)")
    p.add_argument("--violation-only-box", action="store_true",
                  help="If set, do not draw individual rider/vehicle boxes for violations — draw one combined box covering all violations in the image")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as e:
        raise SystemExit("Please install required packages: pip install ultralytics pillow numpy") from e

    images_folder = Path(args.images)
    if not images_folder.exists():
        raise SystemExit(f"Images folder not found: {images_folder}")

    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.weights)
    if not model_path.exists():
        raise SystemExit(f"Weights not found: {model_path}")

    model = YOLO(str(model_path))

    image_files = find_image_files(images_folder)
    if not image_files:
        raise SystemExit(f"No images found in {images_folder}")

    # For CSV report: filename, total_vehicles, total_persons, violation_count
    report_rows: List[Tuple[str, int, int, int]] = []

    for idx, img in enumerate(image_files):
        # Show a single-line progress indicator instead of ultralytics internal progress
        is_tty = sys.stdout.isatty() and not args.no_color
        if not args.quiet and is_tty:
            print(f"Analyzing {idx+1}/{len(image_files)}: {img.name}...", end='\r', flush=True)

        # Suppress ultralytics verbose output (we provide our own progress)
        results = model.predict(source=str(img), conf=args.conf, device=args.device,
                                imgsz=args.imgsz, max_det=args.max_det, save=False, verbose=False)
        r = results[0]  # first batch result

        # Collect persons and vehicles with their boxes
        persons = []
        vehicles = []

        try:
            for box in r.boxes:
                # Get class and convert box coordinates
                if hasattr(box.cls, 'cpu'):
                    cls = int(box.cls.cpu().numpy().item())
                else:
                    cls = int(box.cls)

                # Get bounding box coordinates
                if hasattr(box.xyxy, 'cpu'):
                    xyxy = box.xyxy.cpu().numpy()[0]
                else:
                    xyxy = box.xyxy[0]

                # We assume COCO mapping: 0=person, 3=motorbike
                if cls == 0:  # person
                    persons.append({
                        "bbox": xyxy,
                        "conf": float(box.conf) if hasattr(box.conf, 'item') 
                               else float(box.conf)
                    })
                elif cls == 3:  # motorbike
                    vehicles.append({
                        "bbox": xyxy,
                        "conf": float(box.conf) if hasattr(box.conf, 'item') 
                               else float(box.conf)
                    })
        except Exception as e:
            print(f"Warning: Error processing boxes for {img.name}: {e}")
            continue

        # Group riders with vehicles and detect violations
        vehicle_groups = group_riders_with_vehicles(
            persons, vehicles, 
            iou_threshold=args.iou_thresh,
            distance_threshold=args.dist_thresh
        )

        # Count violations
        violations = sum(1 for g in vehicle_groups if g["is_violation"])

        # Save annotated image with violation markers
        out_path = out_folder / img.name
        annotate_image(str(img), vehicle_groups, out_path,
                       compact_violation=args.compact_violation,
                       show_confidence=args.show_confidence,
                       show_arrow=args.show_arrow,
                       violation_only_box=args.violation_only_box)

        # Print per-image report unless quiet
        if not args.quiet:
            print_detection_report(img.name, vehicle_groups)

        # Add to report
        report_rows.append((
            img.name, 
            len(vehicles),  # total vehicles 
            len(persons),   # total persons
            violations      # number of triple riding violations
        ))

    # When quiet, write CSV-like output and print a minimal summary. Color the status field when colors are enabled.
    use_color = (not args.no_color) and sys.stdout.isatty()

    def _status_text(violations: int) -> str:
        status = "VIOLATION" if violations > 0 else "COMPLIANT"
        if use_color:
            return _color_text(status, 'red' if violations > 0 else 'green')
        return status

    # Write a clean CSV file (plain text) to outputs/<dataset>/report.csv
    try:
        report_file = out_folder.parent / "report.csv"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["Image Name","Total Vehicles","Total Riders","Triple Riding Violations","Status"])
            for row in report_rows:
                filename, vehicles, riders, violations = row
                status = "VIOLATION" if violations > 0 else "COMPLIANT"
                writer.writerow([filename, vehicles, riders, violations, status])
    except Exception as e:
        print(f"Warning: could not write report file: {e}")

    # Minimal final output: do not print CSV or summary to screen by default. Show report location.
    if sys.stdout.isatty():
        print(' ' * 80, end='\r', flush=True)
    print(f"Done. Report written to: {report_file}")


if __name__ == "__main__":
    main()
