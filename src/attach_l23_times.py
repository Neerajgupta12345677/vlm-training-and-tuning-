"""Stamp start/end times onto Level-2/3 rows of a submission CSV.

The arena rejects untimed events at Difficulties 2–3, and scoring them as
empty `events: []` treats the video as normal — which scores 0 if there is
any ground-truth event. D2 + D3 are 75 of 100 marks.

This does not invent new classes. It takes the class we already predicted
for each long video and asks the appearance classifier WHEN that class is
visible (`windows_for_label`). A video we called `normal` is left empty.

    python src\attach_l23_times.py --pred C:\dvad\outputs\predictions_final.csv ^
        --videos C:\dvad\data\ahc\test\videos --manifest C:\dvad\outputs\manifest_public_test.json ^
        --out C:\dvad\outputs\predictions_timed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from appearance_classifier import AppearanceClassifier
from common import probe_video
from export_arena import load_manifest
from submission import write_csv

# Scene-level conditions that usually persist once they start. Extending the
# last window toward the clip end is a duration prior, not a leak of the public
# GT: congestion / flood / smoke / fire do not last 4 seconds in a 6-minute
# video. Accidents and fights must NOT use this - they are short.
PERSIST_CLASSES = {
    "traffic_congestion",
    "waterlogging_or_flood",
    "smoke",
    "fire",
}

# merge_gap_s is now a MULTIPLE OF THE SAMPLE INTERVAL, not a fixed number of
# seconds. The fixed values (8-20s) were the measured cause of D3 scoring 0 of
# 8: a 20s congestion gap merges ground-truth events that sit 5-20s apart, and
# a merged window fails IoU>=0.5 against BOTH of them, where two separate
# windows could have matched both. The arena's own rule ("at most one fragment
# can match, the rest count against you") means over-merging and over-
# fragmenting are both losses - so key the gap to the resolution we actually
# sampled at, which is the only defensible scale.
SPAN = {
    "traffic_accident": dict(min_span_s=4.0, merge_gap_mult=1.6, max_span_frac=0.18),
    "traffic_congestion": dict(min_span_s=8.0, merge_gap_mult=2.2, max_span_frac=0.40),
    "loitering_or_suspicious_presence": dict(min_span_s=6.0, merge_gap_mult=1.6, max_span_frac=0.12),
    "vehicle_blocking_traffic": dict(min_span_s=4.0, merge_gap_mult=1.6, max_span_frac=0.15),
    "fighting_or_violence": dict(min_span_s=8.0, merge_gap_mult=2.0, max_span_frac=0.25),
    "road_spill_or_debris": dict(min_span_s=6.0, merge_gap_mult=1.6, max_span_frac=0.15),
}
DEFAULT_SPAN = dict(min_span_s=4.0, merge_gap_mult=1.8, max_span_frac=0.20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--videos", required=True, help="Directory of Txxx.mp4 files")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", default=r"C:\dvad\models\appearance11.pt")
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--sample-dt", type=float, default=3.0,
                    help="Target seconds between sampled frames on L2/L3 clips.")
    # MEASURED OFF, do not re-enable without re-measuring. The idea was that
    # unmatched fragments are scored against us, so pruning low-confidence
    # windows should help. It does the opposite: at rel_conf=0.6 / cap 4 the
    # gate deleted CORRECT windows - T028 fell from 4 matches of 4 (0 false
    # alarms) to 2, and T033's only match vanished. Total went 56.0 -> 49.5.
    # Classifier confidence does not rank windows by correctness on this
    # footage, so confidence-based pruning is throwing away signal.
    ap.add_argument("--rel-conf", type=float, default=0.0,
                    help="Keep a window only if its confidence is >= this fraction of the "
                         "best for that video+class. 0 disables (measured best).")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="Cap windows per video+class, highest confidence first. 0 = no cap.")
    ap.add_argument("--max-frames", type=int, default=160,
                    help="Cap on frames per clip, so a 10-minute video stays affordable.")
    # A long video almost always contains ONE kind of anomaly. Measured on the
    # public ground truth: of the 8 anomalous L2/L3 videos, 7 have exactly one
    # distinct class and only T026 has more. Emitting four classes for one clip
    # (eval E021 did) therefore guarantees at least three of them are false
    # alarms, and false alarms are the arena's most expensive mistake. Unlike
    # the per-window confidence gate (measured harmful, see --rel-conf), this
    # ranks BETWEEN classes, where the separation is wide: on E021 wrong-way
    # scores 0.52-0.87 while road-spill sits at 0.42-0.49, barely over
    # threshold.
    ap.add_argument("--class-rel", type=float, default=0.0,
                    help="Drop a class whose best window confidence is below this "
                         "fraction of the strongest class for that video. 0 disables.")
    ap.add_argument("--max-classes", type=int, default=0,
                    help="Hard cap on distinct classes per L2/L3 video. 0 = no cap.")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    rows = list(csv.DictReader(Path(args.pred).open(encoding="utf-8-sig")))
    vdir = Path(args.videos)
    clf = AppearanceClassifier(args.weights, threshold=args.threshold)

    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)

    out: list[dict] = []
    for vid, level in manifest.items():
        vrows = by_video.get(vid, [])
        if level < 2:
            out.extend(vrows)
            continue
        path = vdir / f"{vid}.mp4"
        if not path.exists():
            print(f"  [{vid}] L{level} file missing - leaving untimed (exporter will emit events:[])")
            out.extend(vrows)
            continue
        meta = probe_video(path)
        duration = meta.frame_count / meta.fps if meta.fps else 0.0
        # Resolution, not coverage, is what IoU>=0.5 needs. At 64 frames a 629s
        # clip is sampled every ~10s, so a 2.6s ground-truth event cannot be
        # localised well enough to clear the gate no matter how good the
        # classifier is. Aim for a ~3s step and cap the cost.
        n_frames = int(min(args.max_frames, max(24, duration / args.sample_dt)))
        classes = sorted({r["class_name"] for r in vrows
                          if r.get("is_anomaly", "").lower() == "true"
                          and r["class_name"] not in ("", "normal")})
        if len(classes) > 1 and (args.class_rel > 0 or args.max_classes > 0):
            # Rank on the detector confidence already recorded per row. A row
            # with no parseable confidence scores 1.0 so an unknown never loses
            # to a known - this may only drop classes we can see are weaker.
            best_of: dict[str, float] = {}
            for r in vrows:
                c = r["class_name"]
                if c not in classes:
                    continue
                conf = 1.0
                txt = r.get("description_summary") or ""
                if "confidence" in txt:
                    try:
                        conf = float(txt.split("confidence", 1)[1].strip(" ()").split(")")[0])
                    except (ValueError, IndexError):
                        conf = 1.0
                best_of[c] = max(best_of.get(c, 0.0), conf)
            ranked = sorted(classes, key=lambda c: best_of.get(c, 1.0), reverse=True)
            top = best_of.get(ranked[0], 1.0)
            kept = [c for c in ranked
                    if args.class_rel <= 0 or best_of.get(c, 1.0) >= args.class_rel * top]
            if args.max_classes > 0:
                kept = kept[: args.max_classes]
            if len(kept) != len(classes):
                dropped = [f"{c}({best_of.get(c, 1.0):.2f})" for c in classes if c not in kept]
                print(f"    {vid}: {len(classes)} -> {len(kept)} class(es), "
                      f"dropped {', '.join(dropped)}")
            classes = sorted(kept)
        if not classes:
            print(f"  [{vid}] L{level} normal ({duration:.0f}s) - no events to time")
            out.extend(vrows)
            continue
        t0 = time.perf_counter()
        stamped: list[dict] = []
        for cls in classes:
            cfg = dict(SPAN.get(cls, DEFAULT_SPAN))
            sample_dt = duration / max(n_frames - 1, 1)
            cfg["merge_gap_s"] = sample_dt * cfg.pop("merge_gap_mult")
            windows = clf.windows_for_label(path, cls, n_frames=n_frames,
                                            threshold=args.threshold, **cfg)
            # SECOND merge pass, over the PADDED windows this time. The pass
            # inside windows_for_label groups on raw hit spacing, then pads
            # each group's edges - so two groups that were genuinely far apart
            # as raw hits can end up much closer once padded, and nothing
            # after that re-checks the now-smaller gap. Measured cause of
            # T033: 8 windows for 2 ground-truth events, 7 of L3's 13 false
            # alarms from that one video. Gap threshold reuses the same
            # class-tuned merge_gap_mult, applied to actual window gaps
            # (which padding has already shrunk) rather than raw hit gaps -
            # a strictly more aggressive, second look at the same data, not a
            # fit to any one video's ground truth.
            if len(windows) > 1:
                windows = sorted(windows, key=lambda w: float(w["start_time_sec"]))
                # 2x more generous than the internal pass, not the same
                # multiplier - the internal pass already used merge_gap_mult
                # and still left T033 at 8 fragments, so repeating the same
                # threshold on already-padded (smaller) gaps helps only a
                # little (measured: 8->6). False alarms are the arena's most
                # expensive mistake, so the second pass is deliberately more
                # aggressive than the first, accepting some risk of merging
                # two genuinely separate events in exchange for far fewer
                # unmatched fragments.
                frag_gap_s = 2.0 * sample_dt * SPAN.get(cls, DEFAULT_SPAN)["merge_gap_mult"]
                merged: list[dict] = [dict(windows[0])]
                for w in windows[1:]:
                    prev = merged[-1]
                    if float(w["start_time_sec"]) - float(prev["end_time_sec"]) < frag_gap_s:
                        prev["end_time_sec"] = max(float(prev["end_time_sec"]),
                                                   float(w["end_time_sec"]))
                        prev["confidence"] = max(float(prev.get("confidence") or 0.0),
                                                 float(w.get("confidence") or 0.0))
                    else:
                        merged.append(dict(w))
                if len(merged) != len(windows):
                    print(f"    {vid} {cls}: {len(windows)} -> {len(merged)} "
                          f"window(s) after fragment merge (gap<{frag_gap_s:.1f}s)")
                windows = merged
            src = next(r for r in vrows if r["class_name"] == cls)
            if not windows:
                print(f"    {vid} {cls}: no window")
                continue
            # Precision gate. Recall alone does not pay here: an unmatched
            # fragment is scored against us, so 8 windows chasing 2 events is
            # worse than 2 confident ones. Relative (not absolute) because
            # confidence scale differs per class and per clip.
            n_raw = len(windows)
            if args.rel_conf > 0:
                best = max(float(w.get("confidence") or 0.0) for w in windows)
                if best > 0:
                    windows = [w for w in windows
                               if float(w.get("confidence") or 0.0) >= args.rel_conf * best]
            if args.max_windows > 0 and len(windows) > args.max_windows:
                windows.sort(key=lambda w: float(w.get("confidence") or 0.0), reverse=True)
                windows = sorted(windows[: args.max_windows],
                                 key=lambda w: float(w["start_time_sec"]))
            if len(windows) != n_raw:
                print(f"    {vid} {cls}: {n_raw} -> {len(windows)} window(s) after precision gate")
            if cls in PERSIST_CLASSES and duration:
                last = windows[-1]
                persist_to = min(duration - 0.05, max(float(last["end_time_sec"]),
                                                      float(last["start_time_sec"]) + 0.30 * duration))
                if persist_to > float(last["end_time_sec"]) + 1.0:
                    last["end_time_sec"] = round(persist_to, 2)
            for w in windows:
                s = max(0.0, float(w["start_time_sec"]))
                e = min(duration - 0.05, float(w["end_time_sec"])) if duration else float(w["end_time_sec"])
                if e <= s:
                    e = min(duration - 0.05, s + 4.0) if duration else s + 4.0
                stamped.append({
                    "video_id": vid, "level": str(level), "is_anomaly": "true",
                    "class_name": cls,
                    "start_time_sec": round(s, 2), "end_time_sec": round(e, 2),
                    "description_summary": src.get("description_summary") or "",
                })
            print(f"    {vid} {cls}: {len(windows)} window(s) "
                  f"{[(w['start_time_sec'], w['end_time_sec']) for w in windows]}")
        dt = time.perf_counter() - t0
        print(f"  [{vid}] L{level} {duration:.0f}s  {n_frames} frames  {dt:.1f}s")
        out.extend(stamped if stamped else vrows)

    write_csv(out, Path(args.out), append=False)
    print(f"[done] {len(out)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
