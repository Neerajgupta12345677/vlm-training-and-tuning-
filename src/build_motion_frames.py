"""Builds an EGO-COMPENSATED MOTION cache for the temporal anomaly classes.

Why this exists (measured, not speculative)
-------------------------------------------
The appearance classifier over-fires on long test videos - T034: ground truth
says 3% of the clip is anomalous, the classifier fires on 79% of it, a 30x
over-fire. Three measurements pin the cause to one thing:

  1. Training clips are TRIMMED to the event. Median event coverage is 100% for
     7 of 10 classes; `loitering_or_suspicious_presence` yields 810 in-event
     frames and exactly 0 frames of the same scene with nothing happening.
  2. So class correlates perfectly with BACKGROUND APPEARANCE, which is the
     cheapest feature available, and that is what the network learns.
  3. Background is constant within a video, so the per-frame probability is
     near-constant: thresholding is all-or-nothing and confidence ranking
     within a video is noise. (Independently confirmed - ranking the windows by
     confidence put the wrong window first in all 4 L3 videos.)

Localisation is therefore impossible in principle for these classes, at any
threshold and with any window geometry. It is an input problem, not a tuning
problem.

The fix: remove the background from the input.
---------------------------------------------
Each cached sample is a temporal DIFFERENCE image, not a frame. Static scene
content cancels to black, so the network cannot memorise it and is forced onto
motion - which is both the genuinely discriminative signal for these classes
and, unlike appearance, something that actually varies within a video.

Two consequences worth stating plainly:
  * `normal` videos become valid negatives. They are a different scene, which
    made them useless against an appearance model; once the scene is subtracted
    out, "people moving through" vs "a person who stays put" is a like-for-like
    comparison. This is what recovers the negatives the trimmed clips never had.
  * The classes are split by what actually determines them. Fire, smoke,
    waterlogging and road spill ARE appearance - they keep the existing model.
    Only the motion/duration classes come here. This mirrors the rule CLAUDE.md
    already established for the VLM: a single still frame does not contain
    motion.

Ego-motion, and why the naive version fails
-------------------------------------------
This is drone footage. The camera pans, drifts and orbits, so a raw frame
difference is dominated by parallax on the static background and the moving
object is lost in it. Each earlier frame is therefore warped onto the current
frame with a RANSAC partial affine (translation/rotation/uniform scale) fitted
from Shi-Tomasi corners tracked by pyramidal Lucas-Kanade - the same recipe
src\\ego_motion.py already uses for Stage 2. When too few correspondences
survive, the sample is dropped rather than cached as noise.

Three time-scales are differenced into the three channels, because these
classes live at different rates: a fight is fast, a stalled vehicle is slow,
and loitering is only visible against a long baseline.

    python src\\build_motion_frames.py --data_dir C:\\dvad\\data\\ahc ^
        --out C:\\dvad\\data\\motion_frames
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Classes whose definition is motion or duration. A still frame cannot express
# any of these, which is precisely why the appearance model cannot localise
# them. Fire/smoke/waterlogging/road_spill are deliberately absent - they are
# real appearance classes and keep the existing appearance model.
MOTION_CLASSES = [
    "loitering_or_suspicious_presence",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "traffic_congestion",
    "fighting_or_violence",
    "traffic_accident",
]
CLASSES = MOTION_CLASSES + ["normal"]

# Difference baselines, in seconds. Short catches fighting and fast traffic,
# long is what makes a stationary object legible at all: against a 4s baseline
# a loiterer stays black while everything transient around them lights up.
LAGS_S = (0.4, 1.5, 4.0)

# Denser sampling than the appearance cache. Within-video variation is the
# whole point of this input, so it needs to be sampled well enough to show it.
IN_PER_EVENT = 8
OUT_PER_VIDEO = 6
NORMAL_PER_VIDEO = 10

EDGE_MARGIN_S = 1.5

# Differences are small in absolute terms; without gain the cache is almost
# black and JPEG quantisation eats the signal.
GAIN = 3.0

_FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
_FEATURE_PARAMS_RELAXED = dict(maxCorners=600, qualityLevel=0.001, minDistance=5, blockSize=5)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
MIN_CORRESPONDENCES = 12


def load_intervals(train_root: Path) -> dict[str, list[tuple[str, float, float]]]:
    """video stem -> [(class, start, end), ...] for the motion classes only."""
    out: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for gt in sorted(train_root.glob("*/ground_truth.csv")):
        for r in csv.DictReader(gt.open(encoding="utf-8-sig")):
            s, e = (r.get("start_time_sec") or "").strip(), (r.get("end_time_sec") or "").strip()
            cls = (r.get("class_name") or "").strip()
            if not s or not e or cls not in MOTION_CLASSES:
                continue
            try:
                out[(r.get("video_id") or "").strip()].append((cls, float(s), float(e)))
            except ValueError:
                continue
    return dict(out)


def grab(cap, t_sec: float, fps: float, total: int) -> np.ndarray | None:
    idx = int(np.clip(t_sec * fps, 0, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok and fr is not None else None


def warp_onto(prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray | None:
    """Align an earlier frame to the current one, or refuse.

    Returning None on a weak fit is deliberate: a bad warp turns the whole
    background into spurious motion, which is worse than having no sample.
    """
    # Low-contrast aerial scenes (haze, uniform tarmac, dusk) yield too few
    # corners at the default quality bar and were being dropped wholesale -
    # 54% of loitering samples in the smoke test, the one class this cache
    # exists to rescue. Retry once with a lower bar before giving up; a weak
    # corner still constrains a 4-DOF fit, and RANSAC discards it if it lies.
    M = None
    for params in (_FEATURE_PARAMS, _FEATURE_PARAMS_RELAXED):
        pts = cv2.goodFeaturesToTrack(prev_gray, **params)
        if pts is None or len(pts) < MIN_CORRESPONDENCES:
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **_LK_PARAMS)
        if nxt is None or status is None:
            continue
        ok = status.ravel() == 1
        if int(ok.sum()) < MIN_CORRESPONDENCES:
            continue
        M, inliers = cv2.estimateAffinePartial2D(
            pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inliers is None or int(inliers.sum()) < MIN_CORRESPONDENCES:
            M = None
            continue
        break
    if M is None:
        return None
    h, w = cur_gray.shape[:2]
    return cv2.warpAffine(prev_gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def motion_image(cap, t: float, fps: float, total: int, size: int) -> np.ndarray | None:
    """Three ego-compensated difference channels at three time-scales."""
    cur = grab(cap, t, fps, total)
    if cur is None:
        return None
    cur_g = cv2.cvtColor(cv2.resize(cur, (size, size), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    chans = []
    for lag in LAGS_S:
        prev = grab(cap, max(0.0, t - lag), fps, total)
        if prev is None:
            return None
        prev_g = cv2.cvtColor(cv2.resize(prev, (size, size), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2GRAY)
        aligned = warp_onto(prev_g, cur_g)
        if aligned is None:
            return None
        d = cv2.absdiff(cur_g, aligned).astype(np.float32) * GAIN
        chans.append(np.clip(d, 0, 255).astype(np.uint8))
    return cv2.merge(chans)


def gaps_of(evs: list[tuple[str, float, float]], dur: float) -> list[tuple[float, float]]:
    covered = [(max(0.0, s - EDGE_MARGIN_S), min(dur, e + EDGE_MARGIN_S)) for _, s, e in evs]
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(covered):
        if s - cursor > 2.0:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if dur - cursor > 2.0:
        gaps.append((cursor, dur))
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"C:\dvad\data\ahc")
    ap.add_argument("--out", default=r"C:\dvad\data\motion_frames")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="Cap source videos per class folder (0 = no cap).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="This worker's index. Run N processes with --shard 0..N-1.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip samples already cached, so a restart resumes.")
    args = ap.parse_args()

    # Each shard is its own process, so letting OpenCV also fan out internally
    # oversubscribes the CPU and the shards fight each other for cores.
    if args.num_shards > 1:
        cv2.setNumThreads(1)

    train_root = Path(args.data_dir) / "train"
    out_root = Path(args.out)
    intervals = load_intervals(train_root)
    print(f"[gt] {len(intervals)} videos carry motion-class intervals", flush=True)

    for c in CLASSES:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written: dict[str, int] = defaultdict(int)
    dropped = 0

    # Only the motion classes plus `normal`. Sampling fire/smoke folders here
    # would spend minutes building frames no model in this path will consume.
    folders = [f for f in sorted(p.name for p in train_root.iterdir() if p.is_dir())
               if f in MOTION_CLASSES or f == "normal"]

    # Shard across the FLAT video list rather than per folder. Folder sizes run
    # from 4 videos to 632, so handing each worker whole folders would leave one
    # process doing `normal` alone while the rest idle.
    work: list[tuple[str, Path]] = []
    for folder in folders:
        vdir = train_root / folder / "videos"
        if not vdir.exists():
            continue
        videos = sorted(vdir.glob("*.mp4"))
        if args.limit_per_class:
            rng.shuffle(videos)
            videos = sorted(videos[: args.limit_per_class])
        work.extend((folder, v) for v in videos)
    if args.num_shards > 1:
        work = work[args.shard::args.num_shards]
        print(f"[shard {args.shard}/{args.num_shards}] {len(work)} videos", flush=True)

    by_folder: dict[str, list[Path]] = defaultdict(list)
    for folder, v in work:
        by_folder[folder].append(v)

    for folder in folders:
        videos = by_folder.get(folder, [])
        if not videos:
            continue
        n_in = n_out = n_drop = 0

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

            def emit(t: float, cls: str, tag: str) -> bool:
                fp = out_root / cls / f"{folder}__{v.stem}__{tag}.jpg"
                # Sample filenames are deterministic, so an interrupted build
                # resumes instead of paying for the decode + flow twice.
                if args.skip_existing and fp.exists():
                    written[cls] += 1
                    return True
                img = motion_image(cap, float(t), fps, total, args.size)
                if img is None:
                    return False
                cv2.imwrite(str(fp), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                written[cls] += 1
                return True

            if not evs:
                for k, t in enumerate(np.linspace(dur * 0.05, dur * 0.95, NORMAL_PER_VIDEO)):
                    if emit(t, "normal", f"n{k}"):
                        n_out += 1
                    else:
                        n_drop += 1
                cap.release()
                continue

            for ei, (cls, s, e) in enumerate(evs):
                s, e = max(0.0, s), min(dur, e)
                if e <= s:
                    continue
                for k, t in enumerate(np.linspace(s, e, IN_PER_EVENT)):
                    if emit(t, cls, f"e{ei}_{k}"):
                        n_in += 1
                    else:
                        n_drop += 1

            gaps = gaps_of(evs, dur)
            if gaps:
                span = sum(b - a for a, b in gaps)
                for k in range(OUT_PER_VIDEO):
                    x = (k + 0.5) / OUT_PER_VIDEO * span
                    acc, t = 0.0, gaps[0][0]
                    for a, b in gaps:
                        if acc + (b - a) >= x:
                            t = a + (x - acc)
                            break
                        acc += b - a
                    if emit(t, "normal", f"g{k}"):
                        n_out += 1
                    else:
                        n_drop += 1
            cap.release()

        dropped += n_drop
        print(f"  {folder:<34} {len(videos):>4} vids -> {n_in:>5} in-event, "
              f"{n_out:>5} negatives, {n_drop:>5} dropped", flush=True)

    print(f"\n{'class':<34}{'frames':>8}")
    for c in CLASSES:
        print(f"  {c:<32}{written[c]:>8}")
    print(f"\ntotal: {sum(written.values())} frames ({dropped} dropped on weak "
          f"ego-motion fit) -> {out_root}")


if __name__ == "__main__":
    main()
