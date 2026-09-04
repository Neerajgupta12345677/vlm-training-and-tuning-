"""Measure stock YOLO detection quality on real aerial imagery (VisDrone).

Why this matters more than any other benchmark here: Stage 2 and Stage 3 are
scene-agnostic (dwell/zone/neighbour logic is arithmetic), but Stage 1 is a
COCO-trained detector, and COCO is ground-level photography. If a nadir drone
view breaks detection, everything downstream fails no matter how good the
context layer is. So measure recall on genuine aerial data before assuming.

VisDrone classes are collapsed onto the COCO classes we actually use:
  vehicle = car | van | truck | bus        -> COCO car/truck/bus
  person  = pedestrian | people            -> COCO person
  twowheel = bicycle | motor               -> COCO bicycle/motorcycle
Tricycles have no COCO equivalent and are excluded from recall (counted
separately) - they are the closest public proxy to Indian auto-rickshaws.

    python src\\eval_aerial.py --limit 120
    python src\\eval_aerial.py --limit 120 --conf 0.15 --imgsz 1280
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from detect_track import load_detector

VISDRONE_ROOT = Path(r"C:\dvad\data\datasets\VisDrone\VisDrone2019-DET-val")

# VisDrone category id -> our collapsed group
VD_GROUP = {
    1: "person", 2: "person",
    3: "twowheel", 10: "twowheel",
    4: "vehicle", 5: "vehicle", 6: "vehicle", 9: "vehicle",
    7: "tricycle", 8: "tricycle",       # no COCO equivalent
}
# Predicted class NAME -> our collapsed group. Covers both vocabularies so a
# COCO-trained model and a VisDrone-trained model can be compared fairly on the
# same footing: VisDrone emits pedestrian/people/van/motor, none of which exist
# in COCO, and scoring those as misses would unfairly penalise the fine-tune.
NAME_GROUP = {
    # COCO
    "person": "person",
    "bicycle": "twowheel", "motorcycle": "twowheel",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    # VisDrone-only additions
    "pedestrian": "person", "people": "person",
    "van": "vehicle",
    "motor": "twowheel",
    # tricycles have no COCO equivalent; excluded from scoring on both sides
    # so the comparison stays like-for-like
}
COCO_GROUP = NAME_GROUP  # kept as an alias; call sites use NAME_GROUP


def load_gt(ann_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    out = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 6:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        cat = int(parts[5])
        group = VD_GROUP.get(cat)
        if group and w > 0 and h > 0:
            out.append((group, (x, y, x + w, y + h)))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Stock-detector recall on VisDrone aerial imagery.")
    p.add_argument("--root", default=str(VISDRONE_ROOT))
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.3, help="IoU for a GT box to count as found.")
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = p.parse_args()

    root = Path(args.root)
    img_dir, ann_dir = root / "images", root / "annotations"
    if not img_dir.exists():
        raise SystemExit(f"VisDrone val not found at {root}. Download it first.")
    images = sorted(img_dir.glob("*.jpg"))[: args.limit]
    if not images:
        raise SystemExit(f"No images under {img_dir}")

    model, name = load_detector(args.weights, args.device)
    print(f"model={name}  conf={args.conf}  imgsz={args.imgsz}  IoU>={args.iou}")
    print(f"images={len(images)}\n")

    gt_tot, found_tot = Counter(), Counter()
    n_det = 0
    small_found = small_tot = 0

    for i, img_path in enumerate(images, 1):
        ann = ann_dir / (img_path.stem + ".txt")
        if not ann.exists():
            continue
        gts = load_gt(ann)
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = model.predict(frame, verbose=False, conf=args.conf, imgsz=args.imgsz,
                            device=args.device)[0]

        preds = []
        if res.boxes is not None and len(res.boxes):
            n_det += len(res.boxes)
            for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                g = NAME_GROUP.get(str(model.names.get(int(c), "")).lower(), None)
                if g:
                    preds.append((g, tuple(b)))

        diag = float(np.hypot(frame.shape[1], frame.shape[0]))
        for g, box in gts:
            gt_tot[g] += 1
            is_small = np.hypot(box[2] - box[0], box[3] - box[1]) / diag < 0.02
            small_tot += is_small
            hit = any(pg == g and iou(box, pb) >= args.iou for pg, pb in preds)
            if hit:
                found_tot[g] += 1
                small_found += is_small
        if i % 40 == 0:
            print(f"  {i}/{len(images)} images ...")

    print("\n=== recall by group (stock COCO detector on aerial imagery) ===")
    scored = ["vehicle", "person", "twowheel"]
    tot = sum(gt_tot[g] for g in scored)
    hit = sum(found_tot[g] for g in scored)
    for g in scored:
        t, h = gt_tot[g], found_tot[g]
        print(f"  {g:<10} {h:>6}/{t:<6} recall {h/t if t else 0:.3f}")
    print(f"  {'OVERALL':<10} {hit:>6}/{tot:<6} recall {hit/tot if tot else 0:.3f}")
    print(f"\n  tricycles in GT (no COCO class at all): {gt_tot['tricycle']}")
    print(f"  tiny objects (<2% of frame diagonal)  : {small_tot}, "
          f"recall {small_found/small_tot if small_tot else 0:.3f}")
    print(f"  raw detections made                   : {n_det}")
    print("\nRecall is the number that matters here: a missed object can never "
          "become a track, and Stage 2 can only reason about what it is given.")


if __name__ == "__main__":
    main()
