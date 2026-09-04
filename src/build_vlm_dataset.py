"""Builds the VLM fine-tuning set from the organisers' own labels.

This replaces the synthetic n=15 Groq-teacher set the existing LoRA adapter was
trained on. The organisers ship 3173 ground-truth rows carrying a human-written
`description_summary` per training video - real captions, in-distribution with
the private evaluation set. That is strictly better teacher data than anything
we can generate, and it is free.

Design decisions that are NOT arbitrary:
  - FRAMES, NOT VIDEO. Unsloth's UnslothVisionDataCollator does not support
    video tensors (open CUDA device-mismatch bug), so every competitive recipe
    feeds extracted frames as images. We reuse the JPEG cache that
    train_appearance.py already built, so this costs no extra decoding.
  - The target is the SUBMISSION SCHEMA, not free prose. The student is trained
    to emit {is_anomaly, class_name, description_summary} so its output drops
    straight into submission.py with no parsing layer to drift.
  - Frames are grouped per video and one sample is emitted per frame, but the
    train/val split is by VIDEO, so near-duplicate frames of one clip cannot
    straddle the split and inflate the score.

    python src\build_vlm_dataset.py
    python src\build_vlm_dataset.py --max-per-class 200 --frames-per-video 4
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import DATA_DIR

DEFAULT_CACHE = Path(r"C:\dvad\data\appearance_frames")
DEFAULT_LABELS = Path(r"C:\dvad\data\ahc_distill_labels.jsonl")

INSTRUCTION = (
    "You are a drone and CCTV video anomaly analyst. Look at this frame from a "
    "traffic or public-space camera and decide whether it shows an anomaly that "
    "an operator must respond to.\n"
    "Answer with JSON only, using exactly these keys:\n"
    '{"is_anomaly": true|false, "class_name": "<one of: normal, traffic_accident, '
    "traffic_congestion, stalled_or_broken_down_vehicle, vehicle_blocking_traffic, "
    "wrong_way_driving, road_spill_or_debris, waterlogging_or_flood, fire, smoke, "
    'fighting_or_violence, loitering_or_suspicious_presence>", '
    '"description_summary": "<one short sentence describing what you see>"}'
)


def load_labels(path: Path) -> dict[str, dict]:
    """video_id -> label row. The cache encodes the source video stem in each
    filename, which is the same id the organisers use (e.g. TR00335)."""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = (r.get("video_id") or "").strip()
            if vid:
                out[vid] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Unsloth VLM fine-tuning dataset.")
    p.add_argument("--cache", default=str(DEFAULT_CACHE),
                   help="Frame cache written by train_appearance.py.")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    p.add_argument("--out", default=str(DATA_DIR / "vlm_ft"))
    p.add_argument("--frames-per-video", type=int, default=3,
                   help="Cap frames used per video; the cache holds 8.")
    p.add_argument("--max-per-class", type=int, default=400,
                   help="Cap samples per class so `normal` cannot dominate.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy-images", action="store_true", default=True,
                   help="Copy frames next to the JSONL so the folder is self-contained "
                        "and uploadable to Kaggle as one dataset.")
    p.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    args = p.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(
            f"{cache} not found. Build the frame cache first:\n"
            f"  python src\\train_appearance.py --data_dir C:\\dvad\\data\\ahc --extract-only"
        )
    labels = load_labels(Path(args.labels))
    if not labels:
        raise SystemExit(f"No label rows in {args.labels}")

    # Filenames are "<class_folder>__<video_stem>__<k>.jpg", so the TRUE class
    # survives even for clips the appearance classifier lumped into `normal`.
    by_video: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for fp in sorted(cache.rglob("*.jpg")):
        parts = fp.stem.split("__")
        if len(parts) != 3:
            continue
        folder, stem, _ = parts
        by_video[(folder, stem)].append(fp)

    rng = random.Random(args.seed)
    per_class: dict[str, list[tuple[str, list[Path], dict]]] = defaultdict(list)
    missing = 0
    for (folder, stem), frames in by_video.items():
        row = labels.get(stem)
        if row is None:
            missing += 1
            continue
        per_class[row.get("class_name") or folder].append((stem, sorted(frames), row))

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    stats: dict[str, tuple[int, int]] = {}

    for cls in sorted(per_class):
        vids = per_class[cls]
        rng.shuffle(vids)
        if args.max_per_class:
            vids = vids[: args.max_per_class]
        n_val = max(1, int(len(vids) * args.val_frac)) if len(vids) > 1 else 0
        for i, (stem, frames, row) in enumerate(vids):
            target = {
                "is_anomaly": cls != "normal",
                "class_name": cls,
                "description_summary": (row.get("description_summary") or "").strip(),
            }
            for fp in frames[: args.frames_per_video]:
                rel = f"images/{fp.name}"
                if args.copy_images:
                    shutil.copy2(fp, img_dir / fp.name)
                sample = {
                    "image": rel,
                    "video_id": stem,
                    "class_name": cls,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": INSTRUCTION},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": json.dumps(target, ensure_ascii=False)},
                        ]},
                    ],
                }
                (val_rows if i < n_val else train_rows).append(sample)
        stats[cls] = (len(vids) - n_val, n_val)

    for name, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] {out_dir}")
    print(f"  train samples : {len(train_rows)}")
    print(f"  val samples   : {len(val_rows)}")
    if missing:
        print(f"  [warn] {missing} cached video(s) had no matching label row")
    print(f"\n  {'class':<34} {'train_vids':>10} {'val_vids':>9}")
    for cls, (tr, va) in sorted(stats.items()):
        print(f"  {cls:<34} {tr:>10} {va:>9}")
    print("\nNext: upload as a Kaggle dataset, then run the Unsloth notebook on a T4.")


if __name__ == "__main__":
    main()
