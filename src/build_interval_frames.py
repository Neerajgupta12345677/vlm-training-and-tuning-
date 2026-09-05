"""Builds a MOMENT-level frame cache for the appearance classifier.

Measured problem this exists to fix. On the public test set's long videos the
classifier fires far more often than the ground truth says it should:

    video   GT anomalous   classifier fires   over-fire
    T034         3%              79%            30x
    T033        20%              61%             3x
    T032        25%              58%           2.3x

That is not a threshold or window-geometry bug. `train_appearance.py` samples
8-16 frames per training video and labels EVERY ONE of them with the video's
class, including the frames where nothing is happening. So the network is
trained to answer "is this the kind of scene where loitering happens?" and
then asked at inference "is loitering happening at this instant?" - a
different question. On a 357s clip whose scene always looks like a loitering
context, the honest answer to the question it was actually trained on is
"yes, everywhere", which is exactly what it outputs.

Every anomalous training row carries start/end times (verified: 2200 of 2200,
100% coverage across all 11 anomaly classes), so the moment-level labels are
already available and simply were not being used.

Labelling rule:
  frame inside a GT interval          -> that anomaly class
  frame outside every interval,
      in an otherwise anomalous video -> normal   <-- the hard negative
  frame from a `normal` video         -> normal

The middle case is the point. It is the same camera, same scene, same
lighting, differing only in whether the event is happening - which is the
distinction the classifier currently cannot draw. Whole-clip labelling never
produces such a pair, so no amount of retraining on the old cache can teach it.

    python src\\build_interval_frames.py --data_dir C:\\dvad\\data\\ahc ^
        --out C:\\dvad\\data\\interval_frames
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Same eleven anomaly classes the classifier already emits, plus normal.
ANOMALY_CLASSES = [
    "fire", "smoke", "waterlogging_or_flood", "road_spill_or_debris",
    "fighting_or_violence", "traffic_accident",
    "loitering_or_suspicious_presence", "wrong_way_driving",
    "traffic_congestion", "vehicle_blocking_traffic",
    "stalled_or_broken_down_vehicle",
]
CLASSES = ANOMALY_CLASSES + ["normal"]

# Frames sampled INSIDE an event vs OUTSIDE it, per video. The outside frames
# are what teach moment-level discrimination, so they are not an afterthought -
# roughly matching the counts keeps the hard negatives from being drowned out
# while still leaving `normal` videos as the bulk of the negative class.
IN_PER_EVENT = 6
OUT_PER_VIDEO = 6
NORMAL_PER_VIDEO = 8

# A margin around each event so a frame sampled just outside the annotated
# boundary is not labelled `normal` when the event is plainly still on screen.
# Annotation edges are approximate; without this the hardest negatives are
# also the most likely to be mislabelled.
EDGE_MARGIN_S = 1.5


def load_intervals(train_root: Path) -> dict[str, list[tuple[str, float, float]]]:
    """video stem -> [(class, start, end), ...] from every class folder's GT."""
    out: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for gt in sorted(train_root.glob("*/ground_truth.csv")):
        for r in csv.DictReader(gt.open(encoding="utf-8-sig")):
            s, e = (r.get("start_time_sec") or "").strip(), (r.get("end_time_sec") or "").strip()
            if not s or not e:
                continue
            cls = (r.get("class_name") or "").strip()
            if cls not in ANOMALY_CLASSES:
                continue
            try:
                out[(r.get("video_id") or "").strip()].append((cls, float(s), float(e)))
            except ValueError:
                continue
    return dict(out)


def sample_at(cap, t_sec: float, fps: float, total: int) -> np.ndarray | None:
    idx = int(np.clip(t_sec * fps, 0, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok and fr is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"C:\dvad\data\ahc")
    ap.add_argument("--out", default=r"C:\dvad\data\interval_frames")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="Cap source videos per class folder (0 = no cap).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_root = Path(args.data_dir) / "train"
    out_root = Path(args.out)
    intervals = load_intervals(train_root)
    print(f"[gt] {len(intervals)} training videos carry interval annotations")

    for c in CLASSES:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written = defaultdict(int)

    for folder in sorted(p.name for p in train_root.iterdir() if p.is_dir()):
        vdir = train_root / folder / "videos"
        if not vdir.exists():
            continue
        videos = sorted(vdir.glob("*.mp4"))
        if args.limit_per_class:
            rng.shuffle(videos)
            videos = sorted(videos[: args.limit_per_class])
        n_in = n_out = 0

        for v in videos:
            evs = intervals.get(v.stem, [])
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = total / fps if fps else 0.0
            if dur <= 0:
                cap.release()
                continue

            if not evs:
                # A `normal` video: every sampled moment is a negative.
                for k, t in enumerate(np.linspace(dur * 0.05, dur * 0.95, NORMAL_PER_VIDEO)):
                    fr = sample_at(cap, float(t), fps, total)
                    if fr is None:
                        continue
                    fp = out_root / "normal" / f"{folder}__{v.stem}__n{k}.jpg"
                    cv2.imwrite(str(fp), cv2.resize(fr, (args.size, args.size),
                                                    interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    written["normal"] += 1
                    n_out += 1
                cap.release()
                continue

            # Positives: inside each annotated event.
            for ei, (cls, s, e) in enumerate(evs):
                s, e = max(0.0, s), min(dur, e)
                if e <= s:
                    continue
                for k, t in enumerate(np.linspace(s, e, IN_PER_EVENT)):
                    fr = sample_at(cap, float(t), fps, total)
                    if fr is None:
                        continue
                    fp = out_root / cls / f"{folder}__{v.stem}__e{ei}_{k}.jpg"
                    cv2.imwrite(str(fp), cv2.resize(fr, (args.size, args.size),
                                                    interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    written[cls] += 1
                    n_in += 1

            # Hard negatives: same video, moments no event covers.
            covered = [(max(0.0, s - EDGE_MARGIN_S), min(dur, e + EDGE_MARGIN_S))
                       for _, s, e in evs]
            gaps: list[tuple[float, float]] = []
            cursor = 0.0
            for s, e in sorted(covered):
                if s - cursor > 2.0:
                    gaps.append((cursor, s))
                cursor = max(cursor, e)
            if dur - cursor > 2.0:
                gaps.append((cursor, dur))
            if gaps:
                span = sum(b - a for a, b in gaps)
                for k in range(OUT_PER_VIDEO):
                    # Sample uniformly across total gap time, so a long quiet
                    # stretch contributes proportionally more negatives than a
                    # brief one - matching how the test clips are actually shaped.
                    x = (k + 0.5) / OUT_PER_VIDEO * span
                    acc = 0.0
                    t = gaps[0][0]
                    for a, b in gaps:
                        if acc + (b - a) >= x:
                            t = a + (x - acc)
                            break
                        acc += b - a
                    fr = sample_at(cap, float(t), fps, total)
                    if fr is None:
                        continue
                    fp = out_root / "normal" / f"{folder}__{v.stem}__g{k}.jpg"
                    cv2.imwrite(str(fp), cv2.resize(fr, (args.size, args.size),
                                                    interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    written["normal"] += 1
                    n_out += 1
            cap.release()

        print(f"  {folder:<34} {len(videos):>4} videos -> {n_in:>5} in-event, {n_out:>5} negatives")

    print(f"\n{'class':<34}{'frames':>8}")
    for c in CLASSES:
        print(f"  {c:<32}{written[c]:>8}")
    print(f"\ntotal: {sum(written.values())} frames -> {out_root}")


if __name__ == "__main__":
    main()
