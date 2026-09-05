"""Build a VLM fine-tune set from train ground_truth, not clip-level stamps.

The previous builder (`build_vlm_dataset.py`) labelled 3 uniform frames with
the video's folder class. That is right for Level 1 and wrong for a 4-minute
clip whose GT interval is 20 seconds in the middle: the other frames are
taught the anomaly class and the model learns to fire on empty road.

This builder:
  - walks train/<class>/ground_truth.csv (so Saturday is still --data_dir),
  - samples frames INSIDE [start, end] as the event class when timestamps exist,
  - samples a couple of frames OUTSIDE that interval as `normal` when there is
    enough leftover footage (teaches "not every frame of an accident video is
    an accident" - which is what D2/D3 score),
  - uses motion+uniform picking (`frame_sample.sample_window`) instead of
    linspace-only,
  - writes `description_summary` only when it is 20-500 chars (or a truncated
    long caption). Empty/tiny text still trains the class, with a short fallback,
    so the JSON schema stays intact.
  - caps majority classes so `normal` / `traffic_accident` cannot drown fighting.

Does NOT read test/ground_truth.csv. That would leak the public set.

    python src\\build_rich_vlm_dataset.py --data_dir C:\\dvad\\data\\ahc
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from common import DATA_DIR
from frame_sample import _open_video, sample_window

# VLM training does not need 720p/4K JPEGs. Long-edge 768 keeps the event
# readable and cuts decode + disk + later Kaggle image load.
_JPEG_MAX_SIDE = 768

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

# Majority classes get a video cap. Rare ones take every file on disk.
MAJORITY_CAP = {
    "normal": 180,
    "traffic_accident": 180,
    "loitering_or_suspicious_presence": 160,
    "traffic_congestion": 160,
}

# Scarce on disk: 4 stalled videos, 23 congestion, ~80 each for fire/smoke/
# spill. Extra frames from those clips are real variation; a repeated frame is
# not. Spend decode here before falling back to oversampling.
CLASS_FRAMES = {
    "stalled_or_broken_down_vehicle": 24,
    "traffic_congestion": 12,
    "fire": 7,
    "smoke": 7,
    "road_spill_or_debris": 7,
    "waterlogging_or_flood": 6,
}

FALLBACK_DESC = {
    "normal": "Routine activity, no incident visible in this frame.",
    "traffic_accident": "A traffic collision is visible.",
    "traffic_congestion": "Traffic is densely queued and moving slowly.",
    "stalled_or_broken_down_vehicle": "A vehicle is stationary in a live lane.",
    "vehicle_blocking_traffic": "A stationary vehicle is blocking the lane.",
    "wrong_way_driving": "A vehicle is moving against the flow of traffic.",
    "road_spill_or_debris": "Debris or a spill is visible on the roadway.",
    "waterlogging_or_flood": "Flooding or standing water is visible.",
    "fire": "Active fire is visible.",
    "smoke": "A smoke plume is visible.",
    "fighting_or_violence": "A physical fight is visible.",
    "loitering_or_suspicious_presence": "A person is lingering without an obvious purpose.",
}


def _clean_desc(raw: str, cls: str) -> str:
    text = (raw or "").strip()
    if 20 <= len(text) <= 500:
        return text
    if len(text) > 500:
        cut = text[:500]
        # Prefer a sentence boundary so we do not train a clipped clause.
        for sep in (". ", "! ", "? "):
            i = cut.rfind(sep)
            if i >= 40:
                return cut[: i + 1].strip()
        return cut.rstrip() + "."
    return FALLBACK_DESC.get(cls, "An event is visible in this frame.")


def _num(v) -> float | None:
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def iter_train_rows(data_dir: Path) -> list[dict]:
    """One dict per GT row that has a video file on disk."""
    train = data_dir / "train"
    out: list[dict] = []
    for folder in sorted(p for p in train.iterdir() if p.is_dir()):
        gt = folder / "ground_truth.csv"
        if not gt.exists():
            continue
        id_map: dict[str, Path] = {}
        vc = folder / "videos.csv"
        if vc.exists():
            for r in csv.DictReader(vc.open(encoding="utf-8-sig")):
                rel = r.get("filename") or r.get("path") or ""
                vid = r.get("video_id") or Path(rel).stem
                path = folder / rel if rel else folder / "videos" / f"{vid}.mp4"
                id_map[vid] = path
        for r in csv.DictReader(gt.open(encoding="utf-8-sig")):
            vid = r["video_id"]
            path = id_map.get(vid, folder / "videos" / f"{vid}.mp4")
            if not path.exists():
                path = folder / "videos" / Path(path).name
            if not path.exists():
                continue
            out.append({
                "video_id": vid,
                "path": path,
                "class_name": r.get("class_name") or folder.name,
                "start": _num(r.get("start_time_sec")),
                "end": _num(r.get("end_time_sec")),
                "description": r.get("description_summary") or "",
            })
    return out


def _save_jpg(img_dir: Path, name: str, rgb) -> str:
    fp = img_dir / name
    if not fp.exists():
        h, w = rgb.shape[:2]
        m = max(h, w)
        if m > _JPEG_MAX_SIDE:
            scale = _JPEG_MAX_SIDE / m
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"images/{name}"


def _process_one(job: dict) -> dict:
    """Worker: decode one video, write its JPEGs, return sample rows + stats.

    Isolated so ProcessPoolExecutor can pickle it. `job` is a plain dict of
    strings/numbers - no Path objects across the Windows spawn boundary.
    """
    path = Path(job["path"])
    img_dir = Path(job["img_dir"])
    cls, vid = job["class_name"], job["video_id"]
    t0, t1 = job["start"], job["end"]
    desc = job["description"]
    n_in, n_out = job["frames_in"], job["frames_out"]
    samples: list[dict] = []
    stats = {"inside": 0, "outside": 0, "whole": 0, "skip": 0}
    cap, fps, n_total = _open_video(path)
    if cap is not None:
        cap.release()
    if n_total <= 0:
        stats["skip"] = 1
        return {"samples": [], "stats": stats}
    duration = n_total / fps
    has_interval = (
        t0 is not None and t1 is not None and t1 > t0
        and (t1 - t0) < 0.85 * max(duration, 0.1)
        and cls != "normal"
    )
    if has_interval:
        names = [f"{cls}__{vid}__in{k}.jpg" for k in range(n_in)]
        if all((img_dir / n).exists() for n in names):
            for n in names:
                samples.append(_sample(cls, desc, f"images/{n}", vid))
        else:
            for k, (_, fr) in enumerate(sample_window(path, n_in, t0, t1, fps=fps, n_total=n_total)):
                samples.append(_sample(cls, desc, _save_jpg(img_dir, names[k] if k < len(names) else f"{cls}__{vid}__in{k}.jpg", fr), vid))
        stats["inside"] = 1
        if t0 >= 3.0:
            gap = (0.0, t0)
        elif duration - t1 >= 3.0:
            gap = (t1, duration)
        else:
            gap = None
        if gap:
            onames = [f"normal__{vid}__out0{k}.jpg" for k in range(n_out)]
            if all((img_dir / n).exists() for n in onames):
                for n in onames:
                    samples.append(_sample("normal", "", f"images/{n}", vid))
            else:
                for k, (_, fr) in enumerate(sample_window(path, n_out, gap[0], gap[1], fps=fps, n_total=n_total)):
                    samples.append(_sample("normal", "", _save_jpg(img_dir, onames[k] if k < len(onames) else f"normal__{vid}__out0{k}.jpg", fr), vid))
            stats["outside"] = 1
    else:
        n_fr = n_in if duration >= 8.0 else max(3, n_in - 1)
        names = [f"{cls}__{vid}__w{k}.jpg" for k in range(n_fr)]
        if all((img_dir / n).exists() for n in names):
            for n in names:
                samples.append(_sample(cls, desc, f"images/{n}", vid))
        else:
            for k, (_, fr) in enumerate(sample_window(path, n_fr, None, None, fps=fps, n_total=n_total)):
                samples.append(_sample(cls, desc, _save_jpg(img_dir, names[k] if k < len(names) else f"{cls}__{vid}__w{k}.jpg", fr), vid))
        stats["whole"] = 1
    return {"samples": samples, "stats": stats}


def balance(samples: list[dict], target: int, max_repeat: int,
            rng: random.Random) -> list[dict]:
    """Flatten the class histogram: trim the majority, repeat the starved.

    Repeats are capped at `max_repeat` on purpose. A class with 4 source videos
    pumped to full parity teaches those 4 scenes, not the concept, so it is
    better to stay short of target than to memorise.
    """
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_cls[s["class_name"]].append(s)
    out: list[dict] = []
    for cls, items in sorted(by_cls.items()):
        rng.shuffle(items)
        if len(items) >= target:
            out.extend(items[:target])
            continue
        keep = list(items)
        ceiling = min(target, len(items) * max_repeat)
        while len(keep) < ceiling:
            keep.extend(items[: ceiling - len(keep)])
        out.extend(keep)
    rng.shuffle(out)
    return out


def _sample(cls: str, desc: str, image_rel: str, video_id: str) -> dict:
    target = {
        "is_anomaly": cls != "normal",
        "class_name": cls,
        "description_summary": _clean_desc(desc, cls),
    }
    return {
        "image": image_rel,
        "video_id": video_id,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"C:\dvad\data\ahc")
    ap.add_argument("--out", default=str(DATA_DIR / "vlm_ft_rich"))
    ap.add_argument("--frames-in-event", type=int, default=4)
    ap.add_argument("--frames-outside", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel video decodes. 4 is safe on 8GB RAM.")
    ap.add_argument("--target-per-class", type=int, default=400,
                    help="Balanced train count per class. 0 disables balancing.")
    ap.add_argument("--max-repeat", type=int, default=3,
                    help="Cap on oversampling a starved class (see balance()).")
    ap.add_argument("--limit", type=int, default=0, help="Cap videos (0 = all). Debug only.")
    args = ap.parse_args()

    rows = iter_train_rows(Path(args.data_dir))
    if not rows:
        raise SystemExit(f"No on-disk training videos under {args.data_dir}/train")
    rng = random.Random(args.seed)
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cls[r["class_name"]].append(r)

    picked: list[dict] = []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        cap = MAJORITY_CAP.get(cls)
        if cap:
            items = items[:cap]
        picked.extend(items)
    rng.shuffle(picked)
    if args.limit:
        picked = picked[: args.limit]
    print(f"[plan] {len(picked)} videos across {len(by_cls)} classes "
          f"(from {len(rows)} on disk)")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jobs = [{
        "path": str(r["path"]), "img_dir": str(img_dir),
        "class_name": r["class_name"], "video_id": r["video_id"],
        "start": r["start"], "end": r["end"], "description": r["description"],
        "frames_in": CLASS_FRAMES.get(r["class_name"], args.frames_in_event),
        "frames_out": args.frames_outside,
    } for r in picked]
    samples: list[dict] = []
    n_inside = n_outside = n_whole = n_skip = 0
    print(f"[run] {args.workers} workers, {args.frames_in_event} frames/event, "
          f"{args.frames_outside} outside-neg")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                got = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] worker: {e}")
                n_skip += 1
                done += 1
                continue
            samples.extend(got["samples"])
            n_inside += got["stats"]["inside"]
            n_outside += got["stats"]["outside"]
            n_whole += got["stats"]["whole"]
            n_skip += got["stats"]["skip"]
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] samples={len(samples)}  "
                      f"interval={n_inside} outside_neg={n_outside} "
                      f"whole={n_whole} skip={n_skip}")

    # Split by VIDEO, never by frame.
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_vid[s["video_id"]].append(s)
    vids = sorted(by_vid)
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac)) if len(vids) > 1 else 0
    val_ids = set(vids[:n_val])
    train, val = [], []
    for vid, ss in by_vid.items():
        (val if vid in val_ids else train).extend(ss)

    # Balance the train split only. Repeating val rows would just reweight the
    # metric and hide exactly the rare-class failure we are trying to measure.
    n_raw = len(train)
    if args.target_per_class:
        train = balance(train, args.target_per_class, args.max_repeat, rng)

    for name, rows_out in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    seen_img = defaultdict(set)
    for s in train:
        seen_img[s["class_name"]].add(s["image"])
    print(f"\n[ok] {out_dir}")
    print(f"  train samples : {len(train)}  (from {n_raw} before balancing)")
    print(f"  val samples   : {len(val)}")
    print(f"  videos        : {len(by_vid)}  (val {len(val_ids)})")
    print(f"  interval hits : {n_inside}  outside-neg videos: {n_outside}  whole-clip: {n_whole}")
    print("  train class counts (distinct images in brackets):")
    for c, n in Counter(s["class_name"] for s in train).most_common():
        print(f"    {c:<34} {n:>5}  [{len(seen_img[c])}]")


if __name__ == "__main__":
    main()
