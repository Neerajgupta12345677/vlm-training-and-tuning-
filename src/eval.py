"""STAGE 8 - eval harness: quality, latency, and scale estimate.

Three independent evaluations, each optional:

1. Distillation fidelity (--reference + --predictions)
   Student (local VLM) vs teacher (Claude) on the same frames.
   Precision / recall / F1 on `anomalous`, plus severity MAE.

2. Ground-truth detection (--ground-truth + --predictions)
   Against the composited stopped-vehicle clip: did we catch it, how fast,
   and did we fire before the anomaly existed (a true false positive)?

3. Throughput (--run-summary)
   ms/frame and a feeds-per-GPU estimate.

    python src\\eval.py --run-summary C:\\dvad\\outputs\\run_summary.json
    python src\\eval.py --reference C:\\dvad\\data\\pseudo_labels.jsonl --predictions C:\\dvad\\outputs\\events_x.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import OUTPUTS_DIR, read_jsonl


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def eval_distillation(reference: Path, predictions: Path) -> dict:
    """Agreement between the local student and the Claude teacher."""
    ref_rows = [r for r in read_jsonl(reference) if "error" not in r]
    ref = {r["image"]: r for r in ref_rows}
    pred_rows = read_jsonl(predictions)

    tp = fp = fn = tn = 0
    sev_abs_err: list[float] = []
    matched = 0
    disagreements: list[dict] = []

    for p in pred_rows:
        key = p.get("frame_file")
        if key not in ref or not p.get("verdict"):
            continue
        matched += 1
        want = bool(ref[key]["anomalous"])
        got = bool(p["verdict"]["anomalous"])
        sev_abs_err.append(abs(float(ref[key]["severity"]) - float(p["verdict"]["severity"])))
        if got and want:
            tp += 1
        elif got and not want:
            fp += 1
            disagreements.append({"image": key, "teacher": "benign", "student": "ANOMALY",
                                  "teacher_reason": ref[key]["reason"], "student_reason": p["verdict"]["reason"]})
        elif not got and want:
            fn += 1
            disagreements.append({"image": key, "teacher": "ANOMALY", "student": "benign",
                                  "teacher_reason": ref[key]["reason"], "student_reason": p["verdict"]["reason"]})
        else:
            tn += 1

    precision, recall, f1 = prf(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "frames_compared": matched,
        "reference_rows": len(ref_rows),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / total, 3) if total else 0.0,
        "severity_mae": round(sum(sev_abs_err) / len(sev_abs_err), 3) if sev_abs_err else None,
        "disagreements": disagreements[:10],
    }


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def eval_ground_truth(gt_path: Path, predictions: Path, iou_thresh: float = 0.1) -> dict:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    fps = gt.get("fps") or 30.0
    # Rows from a --no-vlm harvesting run carry no verdict; they are candidates,
    # not predictions, so they must not count as detections.
    preds = [p for p in read_jsonl(predictions) if p.get("verdict") and p["verdict"]["anomalous"]]

    results = []
    for anom in gt["anomalies"]:
        start_f = anom["start_frame"]
        gt_box = anom["bbox"]
        after = [p for p in preds if p["frame_idx"] >= start_f]
        # A localised hit needs spatial overlap with the injected vehicle.
        hits = [p for p in after if _iou(p["bbox"], gt_box) >= iou_thresh]
        before = [p for p in preds if p["frame_idx"] < start_f]

        first = min(hits, key=lambda p: p["frame_idx"]) if hits else None
        # Alerts after onset that do NOT overlap the ground-truth object are
        # false positives too. Counting only pre-onset alerts silently ignored
        # every spatially-wrong alert and flattered the result.
        elsewhere = [p for p in after if _iou(p["bbox"], gt_box) < iou_thresh]
        results.append(
            {
                "kind": anom["kind"],
                "detected": bool(hits),
                "gt_start_frame": start_f,
                "first_detection_frame": first["frame_idx"] if first else None,
                "detection_latency_s": round((first["frame_idx"] - start_f) / fps, 2) if first else None,
                "best_iou": round(max((_iou(p["bbox"], gt_box) for p in hits), default=0.0), 3),
                "localised_hits": len(hits),
                "alerts_before_anomaly_existed": len(before),
                "alerts_elsewhere_after_onset": len(elsewhere),
                "false_positives_total": len(before) + len(elsewhere),
                "elsewhere_boxes": [[round(v) for v in p["bbox"]] for p in elsewhere[:5]],
                "severity_at_detection": first["verdict"]["severity"] if first else None,
                "reason_at_detection": first["verdict"]["reason"] if first else None,
            }
        )

    detected = sum(1 for r in results if r["detected"])
    return {
        "ground_truth_file": gt_path.name,
        "anomalies_in_gt": len(results),
        "anomalies_detected": detected,
        "detection_rate": round(detected / len(results), 3) if results else 0.0,
        "total_anomalous_alerts": len(preds),
        "per_anomaly": results,
    }


def eval_throughput(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = data if isinstance(data, list) else [data]
    out = []
    for r in runs:
        out.append(
            {
                "video": r.get("video"),
                "resolution": r.get("resolution"),
                "frames": r.get("frames_processed", r.get("frames")),
                "stride": r.get("stride", 1),
                "stage1_ms_mean": r.get("stage1_ms_mean"),
                "stage1_ms_p95": r.get("stage1_ms_p95"),
                "warm_fps": r.get("warm_fps"),
                "source_fps": r.get("source_fps"),
                "required_fps": r.get("required_fps", r.get("source_fps")),
                "realtime": r.get("realtime"),
                "vlm_calls": r.get("vlm_calls"),
                "vlm_ms_mean": r.get("vlm_ms_mean"),
                "vlm_ms_p95": r.get("vlm_ms_p95"),
                "trigger_rate_pct": r.get("trigger_rate_pct"),
                "feeds_per_gpu_estimate": r.get("feeds_per_gpu_estimate"),
            }
        )
    return {"runs": out}


def main() -> None:
    p = argparse.ArgumentParser(description="Eval harness: F1, latency, feeds-per-GPU.")
    p.add_argument("--predictions", default=None, help="Events .jsonl from pipeline.py.")
    p.add_argument("--reference", default=None, help="Teacher labels .jsonl from distill_label.py.")
    p.add_argument("--ground-truth", default=None, help="*_ground_truth.json from get_sample_data.py.")
    p.add_argument("--run-summary", default=None, help="run_summary.json from pipeline.py.")
    p.add_argument("--data_dir", default=None, help="Unused here; kept for interface parity.")
    p.add_argument("--iou", type=float, default=0.1, help="IoU threshold for a localised hit.")
    p.add_argument("--out", default=None, help="Write the full report JSON here.")
    args = p.parse_args()

    if not any([args.reference, args.ground_truth, args.run_summary]):
        p.print_help()
        raise SystemExit("\nGive at least one of --reference / --ground-truth / --run-summary")

    report: dict = {}

    if args.reference:
        if not args.predictions:
            raise SystemExit("--reference requires --predictions")
        report["distillation"] = eval_distillation(Path(args.reference), Path(args.predictions))
        d = report["distillation"]
        print("=== 1. Distillation fidelity (student vs Claude teacher) ===")
        print(f"  frames compared : {d['frames_compared']} (of {d['reference_rows']} teacher labels)")
        if d["frames_compared"]:
            c = d["confusion"]
            print(f"  confusion       : tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
            print(f"  precision       : {d['precision']}")
            print(f"  recall          : {d['recall']}")
            print(f"  F1              : {d['f1']}")
            print(f"  accuracy        : {d['accuracy']}")
            print(f"  severity MAE    : {d['severity_mae']}")
            for dis in d["disagreements"][:5]:
                print(f"    ! {dis['image'][:40]:<40} teacher={dis['teacher']:<8} student={dis['student']}")
        else:
            print("  [warn] no overlap between predictions and reference (check frame_file names)")
        print()

    if args.ground_truth:
        if not args.predictions:
            raise SystemExit("--ground-truth requires --predictions")
        report["ground_truth"] = eval_ground_truth(Path(args.ground_truth), Path(args.predictions), args.iou)
        g = report["ground_truth"]
        print("=== 2. Ground-truth detection (composited stopped vehicle) ===")
        print(f"  anomalies in GT : {g['anomalies_in_gt']}")
        print(f"  detected        : {g['anomalies_detected']}  (rate {g['detection_rate']})")
        for r in g["per_anomaly"]:
            status = "DETECTED" if r["detected"] else "MISSED"
            print(f"  [{status}] {r['kind']}")
            if r["detected"]:
                print(f"      first alert   : frame {r['first_detection_frame']} "
                      f"(+{r['detection_latency_s']}s after onset)")
                print(f"      best IoU      : {r['best_iou']}")
                print(f"      severity      : {r['severity_at_detection']}")
                print(f"      reason        : {r['reason_at_detection']}")
            print(f"      FALSE POSITIVES : {r['false_positives_total']} "
                  f"({r['alerts_before_anomaly_existed']} before onset, "
                  f"{r['alerts_elsewhere_after_onset']} elsewhere after onset)")
            for b in r["elsewhere_boxes"]:
                print(f"        spurious box: {b}")
        print()

    if args.run_summary:
        report["throughput"] = eval_throughput(Path(args.run_summary))
        print("=== 3. Throughput and scale ===")
        for r in report["throughput"]["runs"]:
            print(f"  {r['video']}  [{r['resolution']}, stride {r['stride']}]")
            print(f"    stage1 mean/p95 : {r['stage1_ms_mean']} / {r['stage1_ms_p95']} ms per frame")
            print(f"    warm FPS        : {r['warm_fps']} vs {r['required_fps']} needed "
                  f"(source {r['source_fps']}fps) -> realtime: {r['realtime']}")
            print(f"    trigger rate    : {r['trigger_rate_pct']}% of frames reached Stage 3")
            print(f"    VLM calls       : {r['vlm_calls']} (mean {r['vlm_ms_mean']} ms, "
                  f"p95 {r['vlm_ms_p95']} ms)")
            print(f"    FEEDS PER GPU   : ~{r['feeds_per_gpu_estimate']} concurrent "
                  f"{r['source_fps']}fps feeds at this resolution")
        print()

    out = Path(args.out) if args.out else OUTPUTS_DIR / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written: {out}")


if __name__ == "__main__":
    main()
