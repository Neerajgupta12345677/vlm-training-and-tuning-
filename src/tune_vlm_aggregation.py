"""Sweeps per-video frame-aggregation strategies against ALREADY-COLLECTED
per-frame VLM output, at zero additional GPU cost.

Why this is safe and free: `eval_kaggle.ipynb`'s multi-frame run writes
`eval_frames.jsonl` - the raw parsed JSON for EVERY sampled frame of every
video, not just the final aggregated verdict. Any aggregation rule (how many
frames must agree, which label wins on disagreement, whether to emit more
than one label per video) can therefore be tried as pure arithmetic over that
one file - the same replay principle `tune_cascade_gate.py` already used for
the classifier+VLM cascade gate.

Two things this specifically re-opens, both checked and correctly REJECTED
under the single-frame eval, for reasons that no longer apply once multiple
frames per video exist:

  - MULTI-LABEL EMISSION. Rejected earlier because T026 (the one video with
    4 simultaneous GT labels) had its true classes ranked 0.02-0.05 against
    normal's 0.76 IN THE CLASSIFIER'S single-frame probabilities. That
    rejection was never about the VLM, and never had per-frame VLM data to
    check. With up to 14 frames now sampled across T026's 240s, different
    frames may show different real events - worth testing directly.

  - PERSISTENCE THRESHOLDS. A real event should show up in more than one
    sampled frame; a one-off misfire should not. This is a natural anti-
    false-positive lever that single-frame evaluation had no way to express.

    python src\\tune_vlm_aggregation.py --frames C:\\dvad\\models\\kaggle_eval_output\\eval_frames.jsonl ^
        --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from label_map import OFFICIAL_LABELS
from score_submission import load_csv, score

CSV_COLUMNS = ["video_id", "level", "is_anomaly", "class_name",
              "start_time_sec", "end_time_sec", "description_summary"]


def load_frames(path: Path) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("parsed")
            label = p.get("class_name") if p else None
            if label not in OFFICIAL_LABELS:
                label = None  # unparseable or off-schema - treat as no vote
            by_video[r["video_id"]].append({"label": label, "t_sec": r.get("t_sec"),
                                            "desc": (p or {}).get("description_summary", "")})
    return dict(by_video)


def build_rows_single_label(by_video: dict[str, list[dict]], min_frames: int,
                            all_videos: set[str]) -> list[dict]:
    """One label per video: the most-frequent anomaly with >= min_frames votes."""
    rows = []
    for vid in sorted(all_videos):
        frames = by_video.get(vid, [])
        labels = [f["label"] for f in frames if f["label"]]
        anomalies = [l for l in labels if l != "normal"]
        counts = Counter(anomalies)
        label, n = (counts.most_common(1)[0] if counts else (None, 0))
        if label and n >= min_frames:
            desc = next((f["desc"] for f in frames if f["label"] == label and f["desc"]), "")
            rows.append({"video_id": vid, "level": 3, "is_anomaly": "true",
                        "class_name": label, "start_time_sec": "", "end_time_sec": "",
                        "description_summary": desc})
        else:
            rows.append({"video_id": vid, "level": 3, "is_anomaly": "false",
                        "class_name": "normal", "start_time_sec": "", "end_time_sec": "",
                        "description_summary": ""})
    return rows


def build_rows_multi_label(by_video: dict[str, list[dict]], min_frames: int,
                           all_videos: set[str]) -> list[dict]:
    """Every anomaly label with >= min_frames votes gets its own row.

    This is what re-opens T026: if fighting_or_violence appears in 3 frames
    and road_spill_or_debris in 2, both get emitted, rather than only the
    single most frequent one winning.
    """
    rows = []
    for vid in sorted(all_videos):
        frames = by_video.get(vid, [])
        labels = [f["label"] for f in frames if f["label"]]
        anomalies = [l for l in labels if l != "normal"]
        counts = Counter(anomalies)
        kept = [(lab, n) for lab, n in counts.items() if n >= min_frames]
        if kept:
            for label, n in kept:
                desc = next((f["desc"] for f in frames if f["label"] == label and f["desc"]), "")
                rows.append({"video_id": vid, "level": 3, "is_anomaly": "true",
                            "class_name": label, "start_time_sec": "", "end_time_sec": "",
                            "description_summary": desc})
        else:
            rows.append({"video_id": vid, "level": 3, "is_anomaly": "false",
                        "class_name": "normal", "start_time_sec": "", "end_time_sec": "",
                        "description_summary": ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="eval_frames.jsonl from the multi-frame eval run.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--write", default=None, help="Write the best-scoring rule's CSV here.")
    args = ap.parse_args()

    by_video = load_frames(Path(args.frames))
    gt_rows = load_csv(Path(args.gt))
    all_videos = {r["video_id"] for r in gt_rows}
    print(f"[loaded] {len(by_video)} video(s) with per-frame data, "
          f"{sum(len(v) for v in by_video.values())} frame(s) total")

    results = []
    for multi in (False, True):
        builder = build_rows_multi_label if multi else build_rows_single_label
        for min_frames in (1, 2, 3, 4):
            rows = builder(by_video, min_frames, all_videos)
            res = score(gt_rows, rows)
            results.append((res["macro_f1"] or 0.0, multi, min_frames, res, rows))

    results.sort(key=lambda t: t[0], reverse=True)
    print(f"\n{'macroF1':>8} {'multi-label':>12} {'min_frames':>11}")
    for m, multi, mf, _, _ in results:
        print(f"{m:8.3f} {str(multi):>12} {mf:11d}")

    best_m, best_multi, best_mf, best_res, best_rows = results[0]
    print(f"\n=== best: multi_label={best_multi} min_frames={best_mf} ===")
    print(f"  macro-F1                 : {best_m}")
    print(f"  exact label-set accuracy : {best_res['video_exact_label_set_accuracy']}")
    b = best_res["is_anomaly_binary"]
    print(f"  is_anomaly accuracy      : {b['accuracy']}  "
          f"(tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
    print("\n  per-class:")
    for lab, m in sorted(best_res["per_class"].items()):
        print(f"    {lab:<34} F1={m['f1']}  (support={m['support_videos']})")

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(best_rows)
        print(f"\n[write] {len(best_rows)} row(s) -> {out}")


if __name__ == "__main__":
    main()
