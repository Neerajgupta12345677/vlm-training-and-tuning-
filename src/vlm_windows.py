"""Turn a per-frame VLM timeline (eval_frames.jsonl) into events with times.

The Kaggle run labels a uniform time grid over each long clip. That grid is a
timeline, so intervals can be rebuilt from it locally - no GPU, instant to
re-tune, and scoreable as many times as needed.

Same three geometry rules the appearance path already uses, for the same
measured reasons:
  * a hit at t only proves the event covers t, so pad each group by half a
    sample interval rather than treating the span of hits as the span of the
    event;
  * merge on a multiple of the ACTUAL sample interval, never a fixed number of
    seconds, or genuinely separate events get fused and the merged window then
    fails an IoU gate against both;
  * grow or truncate about the window CENTRE, so a window is never anchored to
    its first hit.

Two modes. `--mode fill` (default) is the conservative one: it only supplies
videos or classes the existing prediction CSV left empty - which is what the
VLM is uniquely able to do, since it knows classes the 11-class appearance
model deliberately excludes (notably stalled_or_broken_down_vehicle, the cause
of "no window" on eval E022 and E025). `--mode replace` rebuilds every long
video from the timeline instead.

    python src\\vlm_windows.py --frames C:\\dvad\\models\\kaggle_evalframes\\eval_frames.jsonl ^
        --pred C:\\dvad\\outputs\\eval_pred_timed_v3.csv --out C:\\dvad\\outputs\\eval_pred_vlm.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from submission import write_csv

# Same class-conditioned spans the appearance path uses: an accident is short,
# congestion persists.
SPAN = {
    "traffic_accident": dict(min_span_s=4.0, merge_gap_mult=1.6, max_span_frac=0.18),
    "traffic_congestion": dict(min_span_s=8.0, merge_gap_mult=2.2, max_span_frac=0.40),
    "loitering_or_suspicious_presence": dict(min_span_s=6.0, merge_gap_mult=1.6, max_span_frac=0.12),
    "vehicle_blocking_traffic": dict(min_span_s=4.0, merge_gap_mult=1.6, max_span_frac=0.15),
    "stalled_or_broken_down_vehicle": dict(min_span_s=6.0, merge_gap_mult=1.8, max_span_frac=0.20),
    "fighting_or_violence": dict(min_span_s=8.0, merge_gap_mult=2.0, max_span_frac=0.25),
    "road_spill_or_debris": dict(min_span_s=6.0, merge_gap_mult=1.6, max_span_frac=0.15),
    "wrong_way_driving": dict(min_span_s=4.0, merge_gap_mult=1.6, max_span_frac=0.15),
}
DEFAULT_SPAN = dict(min_span_s=4.0, merge_gap_mult=1.8, max_span_frac=0.20)


def build_windows(times: list[float], sample_dt: float, duration: float,
                  cls: str) -> list[tuple[float, float]]:
    """Group hit times into padded, centre-resized intervals."""
    if not times:
        return []
    cfg = SPAN.get(cls, DEFAULT_SPAN)
    gap = sample_dt * cfg["merge_gap_mult"]
    pad = sample_dt / 2.0

    groups: list[list[float]] = [[times[0]]]
    for t in times[1:]:
        if t - groups[-1][-1] <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])

    out: list[tuple[float, float]] = []
    for g in groups:
        start, end = g[0] - pad, g[-1] + pad
        if end - start < cfg["min_span_s"]:
            mid = 0.5 * (start + end)
            start, end = mid - cfg["min_span_s"] / 2.0, mid + cfg["min_span_s"] / 2.0
        max_span = cfg["max_span_frac"] * duration if duration else None
        if max_span and end - start > max_span:
            mid = 0.5 * (start + end)
            start, end = mid - max_span / 2.0, mid + max_span / 2.0
        start = max(0.0, start)
        end = min(duration - 0.05, end) if duration else end
        if end > start:
            out.append((round(start, 2), round(end, 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="eval_frames.jsonl from the Kaggle run")
    ap.add_argument("--pred", required=True, help="Existing prediction CSV to fill or replace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("fill", "replace"), default="fill")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Frames a class needs before it may open a window. 1 is noisy.")
    ap.add_argument("--min-share", type=float, default=0.10,
                    help="Class must hold at least this share of a video's frames.")
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.frames).open(encoding="utf-8") if l.strip()]
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("parsed") and r["parsed"].get("class_name"):
            by_video[r["video_id"]].append(r)

    pred = list(csv.DictReader(Path(args.pred).open(encoding="utf-8-sig")))
    pred_by_video: dict[str, list[dict]] = defaultdict(list)
    for r in pred:
        pred_by_video[r["video_id"]].append(r)

    out: list[dict] = []
    for vid, prows in pred_by_video.items():
        frames = sorted(by_video.get(vid, []), key=lambda f: f.get("t_sec") or 0.0)
        level = int(prows[0].get("level") or 1)
        timed = [r for r in prows if (r.get("start_time_sec") or "").strip()]
        has_events = any(r.get("is_anomaly", "").lower() == "true"
                         and r["class_name"] not in ("", "normal") for r in prows)

        if level < 2 or not frames:
            out.extend(prows)
            continue
        if args.mode == "fill" and timed:
            out.extend(prows)
            continue
        # `fill` on a video whose only prediction is an untimed anomaly is
        # exactly the gap this exists for - the appearance model cannot window
        # a class it was never trained to emit.
        duration = float(frames[0].get("duration_s") or 0.0)
        ts = [f["t_sec"] for f in frames if f.get("t_sec") is not None]
        sample_dt = (max(ts) - min(ts)) / max(len(ts) - 1, 1) if len(ts) > 1 else 5.0

        hits: dict[str, list[float]] = defaultdict(list)
        for f in frames:
            c = f["parsed"]["class_name"]
            if c and c != "normal":
                hits[c].append(float(f["t_sec"]))

        n = len(frames)
        keep = {c: t for c, t in hits.items()
                if len(t) >= args.min_hits and len(t) / max(n, 1) >= args.min_share}
        if not keep:
            print(f"  [{vid}] L{level}: VLM saw nothing persistent "
                  f"({dict(Counter(f['parsed']['class_name'] for f in frames))})")
            out.extend(prows)
            continue

        src = prows[0]
        made = 0
        for cls, times in sorted(keep.items()):
            wins = build_windows(sorted(times), sample_dt, duration, cls)
            for s, e in wins:
                out.append({"video_id": vid, "level": str(level), "is_anomaly": "true",
                            "class_name": cls, "start_time_sec": s, "end_time_sec": e,
                            "description_summary": src.get("description_summary") or ""})
                made += 1
            print(f"    {vid} {cls}: {len(times)}/{n} frames -> {len(wins)} window(s) {wins}")
        print(f"  [{vid}] L{level} {duration:.0f}s  dt={sample_dt:.1f}s  "
              f"{made} event(s){'  (was empty)' if not has_events else ''}")

    write_csv(out, Path(args.out), append=False)
    print(f"[done] {len(out)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
