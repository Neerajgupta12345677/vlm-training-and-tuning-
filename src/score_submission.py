"""Scores a predictions.csv against the organisers' ground_truth.csv.

This is what lets us get a REAL number tonight, on their actual public test
set (34 videos, ~56 min, ships with ground_truth.csv "so teams can validate
their output and scoring pipeline before submitting to the private
evaluation system") - instead of a number self-graded on synthetic footage.

We do not know the private evaluation server's exact metric yet, so this
computes several standard ones and reports all of them rather than betting on
one:
  - video-level exact-match accuracy (does our predicted label SET equal the
    ground-truth label SET for that video, treating it as multi-label)
  - per-class precision/recall/F1 over (video_id, class_name) pairs, plus
    macro-F1 across all 13 classes
  - is_anomaly binary accuracy/precision/recall
  - for (video_id, class_name) pairs present in both and carrying timestamps,
    temporal IoU (mean, and recall at IoU>=0.3 - same threshold convention as
    eval.py's spatial IoU, for consistency across this codebase)

    python src\\score_submission.py --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv --pred C:\\dvad\\outputs\\predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from label_map import OFFICIAL_LABELS

IOU_THRESHOLD = 0.3  # matches eval.py's ground-truth bbox IoU convention


def _parse_bool(s: str) -> bool:
    """Tolerant bool parsing - is_anomaly's on-disk format is genuinely
    unknown until a real ground_truth.csv exists (True/False, true/false,
    TRUE/FALSE, 1/0 are all plausible). Never let a format mismatch here
    silently zero out every row's score."""
    return str(s).strip().lower() in {"true", "1", "yes"}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        missing = {"video_id", "is_anomaly", "class_name"} - set(rows[0].keys())
        if missing:
            raise SystemExit(f"{path} is missing expected column(s): {missing}. "
                             f"Found: {list(rows[0].keys())}")
    return rows


def _interval(row: dict) -> tuple[float, float] | None:
    s, e = row.get("start_time_sec", ""), row.get("end_time_sec", "")
    if s in ("", None) or e in ("", None):
        return None
    try:
        return float(s), float(e)
    except ValueError:
        return None


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], a[0]) - min(a[0], a[1]) + max(b[1], b[0]) - min(b[0], b[1]) - inter
    span_a, span_b = a[1] - a[0], b[1] - b[0]
    union = span_a + span_b - inter
    return inter / union if union > 0 else 0.0


def score(gt_rows: list[dict], pred_rows: list[dict]) -> dict:
    # (video_id, class_name) -> list of rows, since a class can repeat per video.
    gt_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pred_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in gt_rows:
        gt_by_pair[(r["video_id"], r["class_name"])].append(r)
    for r in pred_rows:
        pred_by_pair[(r["video_id"], r["class_name"])].append(r)

    gt_pairs = set(gt_by_pair)
    pred_pairs = set(pred_by_pair)

    # --- per-class precision/recall/F1 over (video, class) SET membership ---
    per_class = {}
    for label in sorted(OFFICIAL_LABELS):
        gt_set = {v for v, c in gt_pairs if c == label}
        pred_set = {v for v, c in pred_pairs if c == label}
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else 0.0)
        support = len(gt_set)
        if support == 0 and tp == fp == 0:
            continue  # class absent from both GT and predictions - not scoreable, don't report
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": round(precision, 3) if precision is not None else None,
                            "recall": round(recall, 3) if recall is not None else None,
                            "f1": round(f1, 3), "support_videos": support}

    scored = [c for c in per_class.values() if c["support_videos"] > 0]
    macro_f1 = round(sum(c["f1"] for c in scored) / len(scored), 3) if scored else None

    # --- video-level exact label-SET match ---
    all_videos = {r["video_id"] for r in gt_rows}
    exact_matches = 0
    for vid in all_videos:
        gt_labels = {c for v, c in gt_pairs if v == vid}
        pred_labels = {c for v, c in pred_pairs if v == vid}
        if gt_labels == pred_labels:
            exact_matches += 1
    video_exact_acc = round(exact_matches / len(all_videos), 3) if all_videos else None

    # --- is_anomaly binary, per GT ROW (a video can have multiple GT rows) ---
    bin_tp = bin_fp = bin_fn = bin_tn = 0
    for gr in gt_rows:
        gt_anom = _parse_bool(gr["is_anomaly"])
        matches = pred_by_pair.get((gr["video_id"], gr["class_name"]), [])
        pred_anom = any(_parse_bool(p["is_anomaly"]) for p in matches) if gt_anom else None
        if gt_anom:
            if matches:
                bin_tp += 1
            else:
                bin_fn += 1
        else:
            # A GT "normal" row: correct only if our prediction for that video
            # contains no anomalous class at all.
            vid_pred_labels = {c for v, c in pred_pairs if v == gr["video_id"] and c != "normal"}
            if vid_pred_labels:
                bin_fp += 1
            else:
                bin_tn += 1
    bin_total = bin_tp + bin_fp + bin_fn + bin_tn
    bin_acc = round((bin_tp + bin_tn) / bin_total, 3) if bin_total else None

    # --- temporal IoU on (video, class) pairs present in both, with timestamps ---
    ious = []
    for pair in gt_pairs & pred_pairs:
        g_ivals = [iv for r in gt_by_pair[pair] if (iv := _interval(r))]
        p_ivals = [iv for r in pred_by_pair[pair] if (iv := _interval(r))]
        if not g_ivals or not p_ivals:
            continue
        # Best-match IoU per GT interval against any predicted interval for
        # the same (video, class) pair.
        for g in g_ivals:
            ious.append(max(_iou_1d(g, p) for p in p_ivals))
    temporal = None
    if ious:
        temporal = {
            "n_matched_intervals": len(ious),
            "mean_iou": round(sum(ious) / len(ious), 3),
            f"recall_at_iou_{IOU_THRESHOLD}": round(sum(1 for i in ious if i >= IOU_THRESHOLD) / len(ious), 3),
        }

    return {
        "videos_scored": len(all_videos),
        "video_exact_label_set_accuracy": video_exact_acc,
        "is_anomaly_binary": {"tp": bin_tp, "fp": bin_fp, "fn": bin_fn, "tn": bin_tn, "accuracy": bin_acc},
        "macro_f1": macro_f1,
        "per_class": per_class,
        "temporal_overlap": temporal,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Score predictions.csv against the organisers' ground_truth.csv.")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    args = p.parse_args()

    gt_rows = load_csv(Path(args.gt))
    pred_rows = load_csv(Path(args.pred))
    print(f"[loaded] {len(gt_rows)} ground-truth row(s), {len(pred_rows)} prediction row(s)")

    result = score(gt_rows, pred_rows)

    print(f"\n=== video-level ===")
    print(f"  videos scored              : {result['videos_scored']}")
    print(f"  exact label-set accuracy   : {result['video_exact_label_set_accuracy']}")
    b = result["is_anomaly_binary"]
    print(f"  is_anomaly accuracy        : {b['accuracy']}  (tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
    print(f"  macro-F1 (13-class)        : {result['macro_f1']}")

    print(f"\n=== per-class (only classes present in GT or predictions) ===")
    for label, m in sorted(result["per_class"].items()):
        print(f"  {label:<32} P={m['precision']}  R={m['recall']}  F1={m['f1']}  "
              f"(support={m['support_videos']} video(s))")

    if result["temporal_overlap"]:
        t = result["temporal_overlap"]
        print(f"\n=== temporal overlap (matched video+class pairs with timestamps) ===")
        print(f"  intervals matched          : {t['n_matched_intervals']}")
        print(f"  mean IoU                   : {t['mean_iou']}")
        print(f"  recall @ IoU>={IOU_THRESHOLD}          : {t[f'recall_at_iou_{IOU_THRESHOLD}']}")
    else:
        print(f"\n=== temporal overlap ===\n  no overlapping (video,class) pairs with timestamps on both sides")


if __name__ == "__main__":
    main()
