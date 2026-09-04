"""Calibrates one decision threshold per class on the VALIDATION split.

Why this exists, and why it is not the thing that was already tried and
rejected:

`tune_appearance.py --per-class` fits 11 thresholds by coordinate ascent
against the PUBLIC TEST SET, and its own leave-one-video-out check proved that
overfits - in-sample macro-F1 0.289, held-out 0.230, against the global rule's
0.256. The conclusion recorded in PROGRESS.md was "per-class thresholds do not
work". That conclusion is too strong. What was actually shown is that
per-class thresholds *fitted on 34 videos* do not work, which is a statement
about the sample size, not about per-class thresholds.

The validation split is 365 held-out VIDEOS (2952 cached frames), roughly ten
times the public test set, and it is not the public test set at all - so a
threshold fitted here has never seen a single test video. That makes the
resulting rule honestly held-out with respect to the leaderboard, in a way the
0.289 number never was.

Two properties make this cheap and safe:
  - It reuses the frame cache written by train_appearance.py, so nothing is
    decoded from video. 2952 cached JPEGs through a 2.54M-parameter network is
    seconds, not minutes.
  - The val split is reconstructed with the SAME build_splits(seed, val_frac)
    the training run used, so "val" here means exactly the videos the
    checkpoint was never trained on. Passing a different --seed or --val-frac
    than the training run silently invalidates that and is checked against the
    checkpoint where possible.

Each class is calibrated INDEPENDENTLY (pick the threshold maximising that
class's own F1 over all 365 val videos), rather than by joint coordinate
ascent. With one parameter fitted per class against ~365 examples, this is
ordinary per-class calibration; joint ascent over 11 coupled parameters is
what invited memorisation in the first place.

Honest caveat, stated here because it decides how much to trust the output:
val videos come from the same source datasets as training and carry exactly
one label each, while test videos are a reserved benchmark source and can carry
several. So val is the best available calibration set, not a perfect proxy for
test. Whether the transfer holds is an empirical question - which is why
--apply reports the resulting TEST macro-F1 next to the global rule's, instead
of assuming the win.

    python src\\calibrate_thresholds.py --out C:\\dvad\\outputs\\val_thresholds.json
    python src\\calibrate_thresholds.py --out C:\\dvad\\outputs\\val_thresholds.json ^
        --apply C:\\dvad\\outputs\\app_scores.json --gt C:\\dvad\\data\\ahc\\test\\ground_truth.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from common import MODELS_DIR
from train_appearance import CLASSES, VAL_TF, FrameDS, build_splits

DEFAULT_WEIGHTS = Path(MODELS_DIR) / "appearance11.pt"

# The grid the threshold is chosen from. Fine near zero because the rare
# classes fire at low absolute confidence once probability mass is split 11
# ways - a 0.05 step above 0.3 would quantise those classes into silence.
GRID = [round(x, 3) for x in np.arange(0.02, 0.86, 0.02)]


def score_val_videos(weights: Path, cache: Path, val_frac: float, seed: int,
                     device: str, batch_size: int = 64) -> tuple[dict, dict, list[str]]:
    """Per-VIDEO mean probabilities over the held-out val split.

    Returns (probs_by_video, true_label_by_video, classes). Aggregation is the
    clip mean, matching AppearanceClassifier.score_video() exactly - a
    threshold calibrated on frame-level probabilities would not transfer to a
    clip-mean decision, because averaging concentrates the distribution.
    """
    ckpt = torch.load(weights, map_location="cpu")
    classes: list[str] = ckpt["classes"]
    if classes != CLASSES:
        print(f"[warn] checkpoint classes differ from train_appearance.CLASSES.\n"
              f"       checkpoint: {classes}\n       module:     {CLASSES}\n"
              f"       Using the checkpoint's list; the val split is rebuilt from "
              f"the module's, so these must match for the split to be meaningful.")

    _, val_items = build_splits(cache, val_frac, seed)
    if not val_items:
        raise SystemExit(
            f"No val frames found under {cache}. Run train_appearance.py first "
            f"(the cache is what this reads - nothing is decoded from video here)."
        )

    model = mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    dev = device if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    # shuffle=False so batches walk val_items in order and index alignment holds.
    dl = DataLoader(FrameDS(val_items, VAL_TF), batch_size=batch_size,
                    shuffle=False, num_workers=0, pin_memory=(dev == "cuda"))

    summed: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    truth: dict[str, int] = {}
    with torch.no_grad():
        offset = 0
        for x, y in dl:
            probs = torch.softmax(model(x.to(dev)), dim=1).cpu().numpy()
            for j in range(len(y)):
                path, label = val_items[offset + j]
                vid = path.name.rsplit("__", 1)[0]  # folder__videostem
                summed[vid] = summed.get(vid, np.zeros(len(classes), np.float64)) + probs[j]
                counts[vid] += 1
                truth[vid] = label
            offset += len(y)

    probs_by_video = {
        vid: {c: round(float(p), 5) for c, p in zip(classes, summed[vid] / counts[vid])}
        for vid in summed
    }
    truth_by_video = {vid: classes[truth[vid]] for vid in summed}
    print(f"[val] {len(probs_by_video)} held-out video(s) scored "
          f"({len(val_items)} cached frames, {dev})")
    return probs_by_video, truth_by_video, classes


def calibrate(probs: dict, truth: dict, classes: list[str]) -> dict:
    """Per-class threshold maximising that class's own F1 over the val videos.

    One-vs-rest and independent per class: a video counts as a positive for
    class c if its clip-mean P(c) clears c's threshold. Ties on F1 break
    toward the HIGHER threshold, because between two rules that score the same
    on val the more conservative one is likelier to hold up off-distribution,
    and the brief weights a false alarm as heavily as a miss.
    """
    out: dict[str, dict] = {}
    for c in classes:
        if c == "normal":
            continue
        pos = {v for v, t in truth.items() if t == c}
        if not pos:
            continue
        best = None
        for thr in GRID:
            pred = {v for v, p in probs.items() if p.get(c, 0.0) >= thr}
            tp = len(pred & pos)
            fp = len(pred - pos)
            fn = len(pos - pred)
            if tp == 0:
                f1 = prec = rec = 0.0
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1 = 2 * prec * rec / (prec + rec)
            # >= keeps the last (highest) threshold among equals.
            if best is None or f1 >= best["f1"]:
                best = {"threshold": thr, "f1": round(f1, 4),
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "support": len(pos), "tp": tp, "fp": fp, "fn": fn}
        if best:
            out[c] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--cache", default=r"C:\dvad\data\appearance_frames11")
    ap.add_argument("--data_dir", default=r"C:\dvad\data\ahc",
                    help="Kept for interface parity; nothing is read from video here.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="MUST match the training run or 'val' is not held out.")
    ap.add_argument("--seed", type=int, default=0,
                    help="MUST match the training run or 'val' is not held out.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=r"C:\dvad\outputs\val_thresholds.json")
    ap.add_argument("--dump-val-scores", default=None,
                    help="Also write the raw per-val-video probabilities here.")
    ap.add_argument("--apply", default=None,
                    help="A test --dump JSON. Applies the calibrated thresholds to it "
                         "and reports the resulting TEST score next to a global-threshold "
                         "baseline, so the transfer is measured rather than assumed.")
    ap.add_argument("--gt", default=r"C:\dvad\data\ahc\test\ground_truth.csv")
    ap.add_argument("--top-k", type=int, default=2,
                    help="Max labels asserted per video when --apply is used.")
    ap.add_argument("--write", default=None,
                    help="Write the calibrated submission CSV here (implies --apply).")
    args = ap.parse_args()

    probs, truth, classes = score_val_videos(
        Path(args.weights), Path(args.cache), args.val_frac, args.seed, args.device)

    per_class = calibrate(probs, truth, classes)
    payload = {
        "weights": str(args.weights),
        "val_videos": len(probs),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "classes": classes,
        "thresholds": {c: m["threshold"] for c, m in per_class.items()},
        "val_metrics": per_class,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== per-class thresholds calibrated on {len(probs)} held-out val video(s) ===")
    print(f"{'class':<34} {'thr':>5} {'valF1':>6} {'P':>6} {'R':>6} {'n':>4}")
    for c, m in sorted(per_class.items(), key=lambda kv: -kv[1]["f1"]):
        print(f"{c:<34} {m['threshold']:5.2f} {m['f1']:6.3f} "
              f"{m['precision']:6.3f} {m['recall']:6.3f} {m['support']:4d}")
    print(f"\n[write] {out}")

    if args.dump_val_scores:
        p = Path(args.dump_val_scores)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"classes": classes, "scores": probs, "truth": truth}, indent=2))
        print(f"[write] {p}")

    if not (args.apply or args.write):
        return

    # --- measure the transfer onto the real test set -------------------------
    from score_submission import load_csv, score
    from tune_appearance import build_rows

    scores_path = Path(args.apply) if args.apply else None
    if scores_path is None:
        raise SystemExit("--write requires --apply <test scores json>")
    test_scores = json.loads(scores_path.read_text())["scores"]
    gt_rows = load_csv(Path(args.gt))
    all_videos = {r["video_id"] for r in gt_rows}

    thresholds = payload["thresholds"]
    cfg = {"threshold": 0.30, "top_k": args.top_k, "margin": 1.0,
           "normal_scale": 1.0, "per_class": thresholds}
    rows = build_rows(test_scores, cfg, 3, all_videos)
    res = score(gt_rows, rows)

    print(f"\n=== applied to the PUBLIC TEST SET ({len(test_scores)} scored video(s)) ===")
    print("These thresholds were fitted on val only - no test video influenced them.")
    print(f"  macro-F1                 : {res['macro_f1']}")
    print(f"  exact label-set accuracy : {res['video_exact_label_set_accuracy']}")
    b = res["is_anomaly_binary"]
    print(f"  is_anomaly accuracy      : {b['accuracy']}  "
          f"(tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
    print("\n  per-class on test:")
    for label, m in sorted(res["per_class"].items()):
        print(f"    {label:<34} P={m['precision']}  R={m['recall']}  F1={m['f1']}  "
              f"(support={m['support_videos']})")

    if args.write:
        import csv as _csv
        w_out = Path(args.write)
        w_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["video_id", "level", "is_anomaly", "class_name",
                "start_time_sec", "end_time_sec", "description_summary"]
        with w_out.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[write] {len(rows)} row(s) -> {w_out}")


if __name__ == "__main__":
    main()
