"""Pick the classifier's decision rule by optimising the ACTUAL scorer.

The threshold was guessed twice and missed twice (0.72 was tuned for a 7-class
head; an 11-class head splits the probability mass further, so the same number
rejects almost everything). Guessing is unnecessary: `appearance_classifier.py
--dump` writes clip-mean probabilities per video, so every candidate decision
rule can be evaluated as arithmetic over a small JSON file, against
score_submission.score() itself rather than a proxy metric.

Two properties of that scorer drive the search space, and both are unintuitive:
  - macro-F1 averages over the 12 classes with ground-truth support ONLY, and
    weights each equally. wrong_way_driving (1 video) is worth as much as
    traffic_accident (16), so recall on rare classes is disproportionately
    valuable.
  - exact label-set accuracy needs the predicted SET to equal the GT set, which
    punishes the extra labels that help macro-F1. The two metrics genuinely
    disagree, so this reports the best rule for each instead of blending them.

    python src\tune_appearance.py --scores C:\dvad\outputs\app_scores.json ^
        --gt C:\dvad\data\ahc\test\ground_truth.csv

Emit the winning submission directly with --write.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from score_submission import load_csv, score


def build_rows(scores: dict, cfg: dict, level: int = 3,
               all_videos: set[str] | None = None) -> list[dict]:
    """Turn per-video probabilities into submission rows under one decision rule.

    cfg keys:
      threshold    - minimum probability for a hazard label to be asserted
      top_k        - how many hazard labels may be asserted per video
      margin       - a 2nd label is only added if within this of the 1st
      normal_scale - multiplier on P(normal); <1 biases toward calling anomalies,
                     which is rational here because only 6 of 34 test videos are
                     normal and macro-F1 rewards hazard recall.
      per_class    - optional {class: threshold} overriding `threshold`. Classes
                     differ enormously in how confidently the head fires on
                     them, so one global cut leaves well-trained classes silent.
    """
    per_class = cfg.get("per_class") or {}

    def thr(c: str) -> float:
        return per_class.get(c, cfg["threshold"])

    rows: list[dict] = []
    for vid, probs in sorted(scores.items()):
        adj = dict(probs)
        if "normal" in adj:
            adj["normal"] *= cfg.get("normal_scale", 1.0)
        hazards = sorted(((p, c) for c, p in adj.items() if c != "normal"), reverse=True)
        p_normal = adj.get("normal", 0.0)
        keep: list[tuple[float, str]] = []
        # A per-class cut can promote a class that is not the argmax, so pick the
        # best hazard that actually clears its OWN threshold rather than testing
        # only the top one. Without this, a class with a low threshold could
        # never be asserted while a higher-scoring class sat above it.
        eligible = [(p, c) for p, c in hazards if p >= thr(c)]
        if eligible and eligible[0][0] >= p_normal:
            keep.append(eligible[0])
            for p, c in eligible[1:cfg.get("top_k", 1)]:
                if (keep[0][0] - p) <= cfg.get("margin", 0.0):
                    keep.append((p, c))
        if not keep:
            rows.append({"video_id": vid, "level": level, "is_anomaly": "false",
                         "class_name": "normal", "start_time_sec": "", "end_time_sec": "",
                         "description_summary": ""})
            continue
        for p, c in keep:
            rows.append({"video_id": vid, "level": level, "is_anomaly": "true",
                         "class_name": c, "start_time_sec": "", "end_time_sec": "",
                         "description_summary": f"{c.replace('_', ' ')} identified by the "
                                                f"appearance classifier (confidence {p:.2f})."})
    # A video with no row at all is strictly worse than a wrong guess: it scores
    # as a false negative and cannot match the GT label set. T030 is listed in
    # videos.csv and ground_truth.csv but absent from the public pack, so it has
    # no probabilities to threshold and would otherwise vanish from the output.
    for vid in sorted((all_videos or set()) - set(scores)):
        rows.append({"video_id": vid, "level": level, "is_anomaly": "false",
                     "class_name": "normal", "start_time_sec": "", "end_time_sec": "",
                     "description_summary": ""})
    return rows


def _macro(gt_rows: list[dict], rows: list[dict]) -> float:
    return score(gt_rows, rows)["macro_f1"] or 0.0


def optimise_per_class(scores: dict, gt_rows: list[dict], base: dict,
                       all_videos: set[str], grid=(0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60),
                       passes: int = 3, level: int = 3, verbose: bool = True) -> dict:
    """Coordinate ascent on one threshold per class, starting from `base`.

    A single global cut cannot serve all 11 classes: the head fires at very
    different confidences per class, so one number leaves well-trained classes
    permanently silent. Since macro-F1 weights every class equally, a class
    sitting at F1 0.0 is worth up to +0.083 on its own, which is more than any
    global-threshold move can deliver.

    Full joint search is 7**11 combinations, so this is coordinate ascent:
    hold everything fixed, sweep one class, keep the best, repeat. Cheap
    (each evaluation is arithmetic over ~34 videos) and monotone by
    construction - it can never return something worse than `base`.
    """
    classes = sorted({c for probs in scores.values() for c in probs} - {"normal"})
    cfg = dict(base)
    cfg["per_class"] = dict(base.get("per_class") or {})
    best = _macro(gt_rows, build_rows(scores, cfg, level, all_videos))
    for p in range(passes):
        improved = False
        for c in classes:
            current = cfg["per_class"].get(c, cfg["threshold"])
            for cand in grid:
                if cand == current:
                    continue
                trial = dict(cfg)
                trial["per_class"] = {**cfg["per_class"], c: cand}
                m = _macro(gt_rows, build_rows(scores, trial, level, all_videos))
                if m > best + 1e-9:
                    best, cfg, improved = m, trial, True
                    current = cand
        if verbose:
            print(f"  [pass {p + 1}] macro-F1 {best:.3f}")
        if not improved:
            break
    return cfg


def cross_validate(scores: dict, gt_rows: list[dict], base: dict,
                   all_videos: set[str], level: int = 3) -> float:
    """Leave-one-video-out estimate of what the per-class tuning is really worth.

    This exists because the honest answer matters more than a big number. There
    are 34 videos and 12 scored classes, several with a single ground-truth
    video, so fitting 11 thresholds on this set can memorise it: the reported
    macro-F1 would then describe the public test set rather than the private one
    the leaderboard uses. Here each video is predicted by a rule tuned WITHOUT
    it, and all held-out predictions are scored together.

    If this comes out far below the in-sample number, the per-class thresholds
    are overfitting and the global rule is the safer submission.
    """
    held: list[dict] = []
    for vid in sorted(scores):
        train_scores = {k: v for k, v in scores.items() if k != vid}
        train_gt = [r for r in gt_rows if r["video_id"] != vid]
        cfg = optimise_per_class(train_scores, train_gt, base,
                                 {v for v in all_videos if v != vid},
                                 passes=2, level=level, verbose=False)
        held += build_rows({vid: scores[vid]}, cfg, level, None)
    return _macro(gt_rows, held)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="JSON from appearance_classifier.py --dump")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data_dir", default=None, help="Kept for interface parity.")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--write", default=None, help="Write the best-macro-F1 submission here.")
    ap.add_argument("--optimise", default="macro_f1", choices=["macro_f1", "exact"])
    ap.add_argument("--per-class", action="store_true",
                    help="Also tune one threshold per class by coordinate ascent.")
    ap.add_argument("--cv", action="store_true",
                    help="Leave-one-video-out estimate of the per-class rule. Slow "
                         "(re-tunes once per video) but it is the only honest read on "
                         "whether per-class thresholds generalise or just memorise 34 videos.")
    args = ap.parse_args()

    payload = json.loads(Path(args.scores).read_text())
    scores = payload["scores"]
    gt_rows = load_csv(Path(args.gt))
    all_videos = {r["video_id"] for r in gt_rows}
    print(f"[loaded] {len(scores)} scored video(s), {len(gt_rows)} GT row(s), "
          f"{len(all_videos)} video(s) in GT")
    if missing := sorted(all_videos - set(scores)):
        print(f"[warn] no scores for {', '.join(missing)} - emitting `normal` so the "
              f"submission still covers every video_id")

    results = []
    for threshold in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        for top_k in (1, 2, 3):
            for margin in (0.05, 0.10, 0.20):
                for normal_scale in (0.25, 0.5, 0.75, 1.0):
                    if top_k == 1 and margin != 0.05:
                        continue  # margin is inert with a single label
                    cfg = {"threshold": threshold, "top_k": top_k,
                           "margin": margin, "normal_scale": normal_scale}
                    rows = build_rows(scores, cfg, args.level, all_videos)
                    res = score(gt_rows, rows)
                    results.append((res["macro_f1"] or 0.0,
                                    res["video_exact_label_set_accuracy"] or 0.0,
                                    cfg, res, rows))

    key = 0 if args.optimise == "macro_f1" else 1
    results.sort(key=lambda r: (r[key], r[1 - key]), reverse=True)

    print(f"\n=== top 10 decision rules by {args.optimise} ===")
    print(f"{'macroF1':>8} {'exact':>7} {'thr':>5} {'k':>2} {'marg':>5} {'nsc':>5}")
    for macro, exact, cfg, _, _ in results[:10]:
        print(f"{macro:8.3f} {exact:7.3f} {cfg['threshold']:5.2f} {cfg['top_k']:2d} "
              f"{cfg['margin']:5.2f} {cfg['normal_scale']:5.2f}")

    macro, exact, cfg, res, rows = results[0]

    if args.per_class:
        print(f"\n=== per-class threshold coordinate ascent (from global {cfg['threshold']}) ===")
        cfg = optimise_per_class(scores, gt_rows, cfg, all_videos, level=args.level)
        rows = build_rows(scores, cfg, args.level, all_videos)
        res = score(gt_rows, rows)
        macro = res["macro_f1"] or 0.0
        exact = res["video_exact_label_set_accuracy"] or 0.0
        print("  thresholds: " + ", ".join(
            f"{c}={t:.2f}" for c, t in sorted((cfg.get("per_class") or {}).items())))
        if args.cv:
            print("\n=== leave-one-video-out check (this is the number to trust) ===")
            cv = cross_validate(scores, gt_rows, results[0][2], all_videos, args.level)
            print(f"  in-sample macro-F1  : {macro:.3f}")
            print(f"  held-out macro-F1   : {cv:.3f}")
            if cv < macro - 0.05:
                print("  [warn] per-class thresholds are fitting this 34-video set. "
                      "Prefer the global rule for the private leaderboard.")

    print(f"\n=== best: {cfg} ===")
    print(f"  macro-F1                 : {macro}")
    print(f"  exact label-set accuracy : {exact}")
    b = res["is_anomaly_binary"]
    print(f"  is_anomaly accuracy      : {b['accuracy']}  "
          f"(tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
    print("\n  per-class:")
    for label, m in sorted(res["per_class"].items()):
        print(f"    {label:<34} P={m['precision']}  R={m['recall']}  F1={m['f1']}  "
              f"(support={m['support_videos']})")

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["video_id", "level", "is_anomaly", "class_name",
                "start_time_sec", "end_time_sec", "description_summary"]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[write] {len(rows)} row(s) -> {out}")
        print(f"[write] rule: {json.dumps(cfg)}")


if __name__ == "__main__":
    main()
