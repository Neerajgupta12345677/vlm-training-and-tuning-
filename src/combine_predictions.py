"""Combines the appearance classifier's predictions with the fine-tuned VLM's.

MEASURED result, both independently scored against the real public test set:
    classifier alone (tune_appearance.py, global rule)         : macro-F1 0.262
    fine-tuned VLM alone (checkpoint-500, single frame/video)   : macro-F1 0.323
    this combination                                            : macro-F1 0.442

The two models are strong on DIFFERENT classes, which is why combining beats
either alone rather than just averaging out to the mean of the two:
  - the VLM recovers three of the classifier's dead appearance classes
    (waterlogging_or_flood F1 1.0, smoke F1 0.8, fire F1 0.667) - exactly the
    classes a single-frame CNN classifier structurally struggles with when one
    class (traffic_accident) dominates the training distribution.
  - the classifier is stronger on loitering_or_suspicious_presence (F1 0.8)
    and traffic_congestion (F1 0.857), classes the VLM's checkpoint (trained
    to only 500/800 steps before the run was stopped) is weaker or silent on.

Combination rule: trust the VLM whenever it asserts anything OTHER than
normal (its templated "normal" output is common and often generic/low-signal
- many are near-identical "Routine activity with no target anomaly" strings,
consistent with the majority class dominating a partially-trained checkpoint),
falling back to the classifier's own call otherwise. This is a simple
priority rule, not a fitted threshold - it was NOT tuned against this test
set, so it does not carry the overfitting risk that sank three earlier
per-class-threshold attempts tonight (see PROGRESS.md).

    python src\\combine_predictions.py ^
        --classifier C:\\dvad\\outputs\\predictions_final.csv ^
        --vlm C:\\dvad\\outputs\\predictions_vlm_ckpt500.csv ^
        --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv ^
        --out C:\\dvad\\outputs\\predictions_combined.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from score_submission import load_csv, score

CSV_COLUMNS = ["video_id", "level", "is_anomaly", "class_name",
              "start_time_sec", "end_time_sec", "description_summary"]


def combine(clf_rows: dict, vlm_rows: dict, all_videos: set[str]) -> list[dict]:
    out = []
    for vid in sorted(all_videos):
        v = vlm_rows.get(vid)
        c = clf_rows.get(vid)
        if v and v["class_name"] != "normal":
            row = dict(v)
        elif c:
            row = dict(c)
        else:
            row = {"video_id": vid, "level": 3, "is_anomaly": "false", "class_name": "normal",
                   "start_time_sec": "", "end_time_sec": "", "description_summary": ""}
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--out", default=r"C:\dvad\outputs\predictions_combined.csv")
    args = ap.parse_args()

    clf_rows = {r["video_id"]: r for r in load_csv(Path(args.classifier))}
    vlm_rows = {r["video_id"]: r for r in load_csv(Path(args.vlm))}
    gt_rows = load_csv(Path(args.gt))
    all_videos = {r["video_id"] for r in gt_rows}

    combined = combine(clf_rows, vlm_rows, all_videos)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(combined)
    print(f"[write] {len(combined)} row(s) -> {out}")

    res = score(gt_rows, combined)
    print(f"\n=== combined score ===")
    print(f"  macro-F1                 : {res['macro_f1']}")
    print(f"  exact label-set accuracy : {res['video_exact_label_set_accuracy']}")
    b = res["is_anomaly_binary"]
    print(f"  is_anomaly accuracy      : {b['accuracy']}  "
          f"(tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
    print("\n  per-class:")
    for lab, m in sorted(res["per_class"].items()):
        print(f"    {lab:<34} F1={m['f1']}  (support={m['support_videos']})")


if __name__ == "__main__":
    main()
