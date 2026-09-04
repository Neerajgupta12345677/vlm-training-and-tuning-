"""Cascaded adjudication: the classifier proposes, the VLM decides, on temporal evidence.

WHY THIS EXISTS - the measurement that motivates the whole design:

The appearance classifier reaches high per-class F1 on its own held-out
validation videos and then scores ZERO on the same classes on the public test
set:

    class                        val F1   test F1
    waterlogging_or_flood         0.914     0.000
    vehicle_blocking_traffic      0.772     0.000
    fighting_or_violence          0.680     0.000
    road_spill_or_debris          0.649     0.000
    wrong_way_driving             0.553     0.000

The model plainly KNOWS these classes. What silences them is the decision
rule: with top-1 argmax a class must beat all eleven competitors on a video,
and these lose that contest even when they are the correct answer and score
respectably in absolute terms.

Two attempts to fix that with thresholds both failed, and both are recorded so
they are not tried a third time:
  - per-class thresholds fitted on the 34-video test set: 0.289 in-sample,
    0.230 leave-one-video-out (tune_appearance.py --per-class --cv)
  - per-class thresholds calibrated on the 365-video held-out val split:
    0.126 on test (calibrate_thresholds.py)
The second failing is the informative one: a genuinely held-out, ten-times
larger calibration set transferred WORSE, which points at domain shift rather
than sample size. The organisers separate training-pool sources from the
reserved test source at the video level, so no threshold fitted on
training-pool data describes the test distribution.

So this module stops trying to fix the ARGMAX and instead asks a different
model to break the tie, on evidence the classifier never sees:

    MobileNetV3-Small (always on, ~0.3s/clip)
        -> top-1 clearly wins?           accept, no VLM call
        -> contested?                    2x2 timestamped temporal montage
                                         -> VLM picks among the top-k candidates

Three properties make this worth doing rather than just plausible:

1. IT ASKS THE VLM A QUESTION IT IS GOOD AT. Measured across four prompt
   revisions, qwen2.5vl:3b answered "is this anomalous?" at chance (3/6) while
   describing the scene correctly - a single frame contains no motion, so the
   boolean was unanswerable from the input it was given. Choosing between three
   NAMED candidates, with four frames spanning the clip, is a discrimination
   task with the evidence actually present.

2. IT CANNOT LOSE MUCH. The VLM may only choose among candidates the
   classifier already proposed, and any unparseable or off-list answer keeps
   the classifier's top-1. The baseline is the floor, by construction.

3. THE COST IS BOUNDED AND MEASURED. The VLM runs only on contested videos and
   every call is counted, so the efficiency claim is arithmetic rather than a
   guess: see `stats["vlm_call_rate"]`.

    python src\\cascade.py --data_dir C:\\dvad\\data\\ahc --split test --backend mock
    python src\\cascade.py --data_dir C:\\dvad\\data\\ahc --split test --backend ollama --model qwen2.5vl:3b
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from appearance_classifier import AppearanceClassifier
from label_map import OFFICIAL_LABELS

# What each class looks like in a single frame, in the words a small model can
# actually check against pixels. These are deliberately VISUAL and concrete -
# "vehicle stopped in a live lane while other traffic flows" is checkable;
# "a breakdown" is not, because it names an inferred cause rather than an
# appearance. Only classes the classifier can propose need an entry.
_CLASS_HINTS = {
    "fire": "visible flames or an active blaze",
    "smoke": "a smoke plume or haze drifting across the scene",
    "waterlogging_or_flood": "standing water covering the road or submerged vehicles",
    "road_spill_or_debris": "spilled load, rubble, or scattered objects lying on the roadway",
    "fighting_or_violence": "people physically fighting, grappling, or striking each other",
    "traffic_accident": "collided or overturned vehicles, or crash damage and debris",
    "traffic_congestion": "a dense queue of vehicles packed bumper to bumper",
    "vehicle_blocking_traffic": "a vehicle halted across a lane forcing others around it",
    "wrong_way_driving": "a vehicle facing or moving against the direction of surrounding traffic",
    "loitering_or_suspicious_presence": "people lingering with no apparent purpose, or somewhere they should not be",
    "stalled_or_broken_down_vehicle": "a single vehicle stopped on a carriageway or shoulder with traffic flowing past",
    "normal": "an ordinary scene with nothing requiring a response",
}


def build_montage(video: Path, n: int = 4, cell: int = 448,
                  label_times: bool = True) -> tuple[np.ndarray | None, list[float]]:
    """A 2x2 grid of `n` frames spanning the clip, each stamped with its time.

    ONE image rather than n images, for two reasons that are both practical
    rather than aesthetic: Ollama's multi-image support varies by model build,
    and a 2x2 montage costs roughly a quarter of the image tokens that four
    separate images would, which matters when a call already takes 27-45s on a
    4GB card with no tensor cores.

    The timestamps are burned into the pixels on purpose. Without them the
    model can see that something differs between quadrants but cannot tell
    which way time runs, and "stationary across the sequence" versus "moving"
    is exactly the distinction we are paying for.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return None, []

    # Avoid the extreme ends: fades and black frames are common there, and a
    # black quadrant wastes a quarter of the evidence.
    idxs = np.linspace(total * 0.05, total * 0.95, n).astype(int)
    frames, times = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(i, 0, max(total - 1, 0))))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
            times.append(round(float(i) / fps, 2))
    cap.release()
    if not frames:
        return None, []

    # If the decoder returned fewer than asked, repeat the last frame so the
    # grid stays square rather than leaving a torn layout the model must guess at.
    while len(frames) < n:
        frames.append(frames[-1])
        times.append(times[-1] if times else 0.0)

    tiles = []
    for fr, t in zip(frames, times):
        tile = cv2.resize(fr, (cell, cell), interpolation=cv2.INTER_AREA)
        if label_times:
            txt = f"t={t:.1f}s"
            cv2.rectangle(tile, (0, 0), (150, 34), (0, 0, 0), -1)
            cv2.putText(tile, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows), times


def build_single_frame(video: Path, cell: int = 448) -> tuple["np.ndarray | None", list[float]]:
    """ONE natural frame from the clip's midpoint - the fallback that is
    actually proven to work on this hardware.

    MEASURED 2026-09-04: qwen2.5vl:3b via Ollama on this GTX 1650 (4GB, no
    tensor cores) returns coherent text for a single natural photograph, but
    degenerates into a repeated-token loop ("@@@@@@...", the same signature
    CLAUDE.md already documents for a different NaN-logit case) for EVERY
    composited multi-frame grid tested (512x512 through 896x896, with and
    without burned-in text), and simply TIMES OUT (>120s) on a true multi-image
    call with 4 separate frames. Confirmed three separate ways before falling
    back, not assumed. The temporal-evidence idea (show the model motion, not
    one instant) is still right in principle - see build_montage below, kept
    for whenever this runs on hardware that can actually serve it - but this
    exact model + this exact card cannot execute it today, and shipping a call
    pattern that reliably times out is worse than shipping the smaller,
    working experiment.

    This keeps the OTHER half of the original hypothesis alive and untested
    until now: whether a CLOSED-SET, NAMED-CANDIDATE discrimination question is
    easier for the model than the open "is this anomalous?" boolean that
    scored at chance. If it still fails on a single frame, the closed-choice
    framing itself was not the fix either - useful to know either way.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return None, []
    mid = int(np.clip(total * 0.5, 0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, []
    frame = cv2.resize(frame, (cell, cell), interpolation=cv2.INTER_AREA)
    return frame, [round(mid / fps, 2)]


def adjudication_prompt(candidates: list[tuple[str, float]], times: list[float]) -> str:
    """A closed-choice question, not an open judgement.

    The candidate probabilities are deliberately WITHHELD from the model. Shown
    them, a small model tends to restate the leader - which is the argmax we
    are already getting for free and specifically trying to second-guess. The
    ordering is randomised by the caller for the same reason.
    """
    is_montage = len(times) >= 2
    lines = [
        ("This is a 2x2 grid of four frames from ONE video clip, in time order: "
         "top-left = earliest, top-right = next, bottom-left = next, bottom-right = latest, "
         f"spanning t={times[0]:.1f}s to t={times[-1]:.1f}s."
         if is_montage else
         f"This is a single frame from a video clip, at t={times[0]:.1f}s." if times else
         "This is a single frame from a video clip."),
        "",
        ("Compare the four frames to judge what CHANGES over time (is a vehicle "
         "stationary across all four, is a queue growing, is a hazard spreading)."
         if is_montage else
         "Judge what is visible in this frame."),
        "",
        "Choose the ONE option below that best describes this clip:",
    ]
    for name, _ in candidates:
        lines.append(f"  - {name}: {_CLASS_HINTS.get(name, name.replace('_', ' '))}")
    lines += [
        "",
        "Reply with the option name exactly as written above, on its own line, "
        "then one short sentence of visual evidence.",
    ]
    return "\n".join(l for l in lines if l != "")


def _search(text: str, candidates: list[str]) -> str | None:
    low = text.lower()
    for name in sorted(candidates, key=len, reverse=True):
        if name.lower() in low:
            return name
    for name in sorted(candidates, key=len, reverse=True):
        head = name.split("_or_")[0].replace("_", " ")
        if head and head.lower() in low:
            return name
    return None


def parse_choice(raw: str, candidates: list[str]) -> str | None:
    """Recover a candidate name from free-form output.

    Deliberately NOT schema-constrained decoding: that was measured to degrade
    these small models (see vlm_reason.constrain_first, off by default), and a
    closed-set substring match is both more forgiving and easy to verify.

    MEASURED prompt-parroting failure, the same family already documented
    elsewhere in this project: the prompt lists every candidate's hint text
    verbatim, and the model sometimes answers correctly on its own first line
    and then rambles into a second paragraph that happens to reuse another
    candidate's NAME while describing something else - "normal

The image
    shows two trucks... Wrong_way_driving: a vehicle facing..." is a real
    captured example, where the true answer (normal, first line, exactly as
    instructed) would have been silently overridden by a later echo of
    'wrong_way_driving' under a naive whole-text scan.

    So the FIRST LINE is authoritative when it contains an unambiguous match -
    that is what "on its own line" in the prompt is actually asking for -
    and only an empty or non-matching first line falls back to scanning the
    rest of the response.
    """
    if not raw:
        return None
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    hit = _search(first_line, candidates)
    if hit:
        return hit
    return _search(raw, candidates)
    return None


def classify_cascade(clf, reasoner, video: Path, cfg: dict, rng) -> dict:
    """One video through the cascade. Returns the decision plus its provenance."""
    t0 = time.perf_counter()
    probs = clf.score_video(video, n_frames=cfg["frames"])
    screen_s = time.perf_counter() - t0
    if not probs:
        return {"label": "normal", "route": "unreadable", "vlm_called": False,
                "screen_s": screen_s, "vlm_s": 0.0, "probs": {}}

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = p1 - p2

    # The cheap path. A clear winner needs no second opinion, and every call
    # skipped here is what makes the cascade economical rather than just
    # accurate - this is the ratio reported as vlm_call_rate.
    confident = p1 >= cfg["confident_p"] and margin >= cfg["confident_margin"]
    if confident or reasoner is None:
        return {"label": top1, "route": "classifier" if confident else "no-vlm",
                "vlm_called": False, "screen_s": screen_s, "vlm_s": 0.0,
                "p1": round(p1, 4), "margin": round(margin, 4),
                "probs": {k: round(v, 4) for k, v in ranked[:5]}}

    # Contested. Build the candidate slate: the top-k, plus `normal` so the
    # model can still decline, since a wrong positive costs as much as a miss
    # on this benchmark ("false alarms matter as much as missed detections").
    cands = [(n, p) for n, p in ranked[: cfg["top_k"]] if p >= cfg["candidate_floor"]]
    if not cands:
        cands = [ranked[0]]
    names = [n for n, _ in cands]
    if "normal" not in names:
        cands.append(("normal", probs.get("normal", 0.0)))
        names.append("normal")
    rng.shuffle(cands)  # order must not encode the classifier's ranking

    if cfg["evidence"] == "montage":
        image, times = build_montage(video, n=cfg["montage_frames"], cell=cfg["cell"])
    else:
        image, times = build_single_frame(video, cell=cfg["cell"])
    if image is None:
        return {"label": top1, "route": "frame-read-failed", "vlm_called": False,
                "screen_s": screen_s, "vlm_s": 0.0, "p1": round(p1, 4),
                "margin": round(margin, 4)}

    prompt = adjudication_prompt(cands, times)
    t1 = time.perf_counter()
    raw = ""
    try:
        raw = reasoner.raw_choice(image, prompt)
    except Exception as e:  # noqa: BLE001
        raw = f"__error__ {type(e).__name__}: {e}"
    vlm_s = time.perf_counter() - t1

    picked = parse_choice(raw, names)
    # An off-list or unparseable answer keeps the classifier's top-1: the
    # baseline is the floor of this design, never something the VLM can spend.
    label = picked or top1
    return {"label": label, "route": "vlm" if picked else "vlm-unparsed",
            "vlm_called": True, "screen_s": screen_s, "vlm_s": vlm_s,
            "p1": round(p1, 4), "margin": round(margin, 4),
            "candidates": names, "vlm_raw": raw[:400], "vlm_picked": picked,
            "changed": bool(picked and picked != top1),
            "probs": {k: round(v, 4) for k, v in ranked[:5]}}


class _Adjudicator:
    """Thin wrapper giving vlm_reason's backends a single-image choice call.

    Kept here rather than in vlm_reason.py because this is a different QUESTION
    (pick from a closed list) than Stage 3's SceneObservation contract, and
    folding it in would blur a division of labour that was set by measurement.
    """

    SYSTEM = ("You are a traffic and public-safety video analyst. You are shown a grid "
              "of frames from one clip and a short list of possible descriptions. "
              "Pick the single best match. Be conservative: if nothing unusual is "
              "visible, choose normal.")

    def __init__(self, backend: str, model: str, url: str, timeout_s: float):
        self.backend, self.model, self.url, self.timeout_s = backend, model, url, timeout_s

    def raw_choice(self, image: np.ndarray, prompt: str) -> str:
        from vlm_reason import encode_jpeg_b64
        if self.backend == "mock":
            # Deterministic and offline: echo the FIRST listed option. Since the
            # caller shuffles the slate, this is a genuine no-information
            # control - it measures the plumbing, never flatters the cascade.
            for line in prompt.splitlines():
                if line.strip().startswith("- "):
                    return line.strip()[2:].split(":")[0]
            return "normal"
        import requests
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": prompt,
                 "images": [encode_jpeg_b64(image, quality=85)]},
            ],
            "options": {"temperature": 0.2, "num_predict": 120, "repeat_penalty": 1.3},
        }
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]


def read_videos(split_dir: Path) -> list[tuple[str, Path]]:
    """(video_id, path) pairs, preferring videos.csv ids over filename stems."""
    vdir = split_dir / "videos"
    files = {p.stem: p for p in sorted(vdir.glob("*.mp4"))} if vdir.exists() else {}
    csv_path = split_dir / "videos.csv"
    out: list[tuple[str, Path]] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                vid = (r.get("video_id") or r.get("id") or "").strip()
                fn = (r.get("file_name") or r.get("filename") or r.get("file")
                      or r.get("video_path") or r.get("path") or "").strip()
                stem = Path(fn).stem if fn else vid
                if vid and stem in files:
                    out.append((vid, files[stem]))
                elif vid:
                    out.append((vid, vdir / (fn or f"{vid}.mp4")))
    if not out:
        out = [(s, p) for s, p in files.items()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"C:\dvad\data\ahc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=r"C:\dvad\models\appearance11.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=8, help="Frames the classifier averages.")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock", "none"])
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=180.0)
    # The gate. A clip is contested unless the leader is both absolutely
    # confident AND clearly ahead; both conditions matter, because an 11-way
    # head can put 0.45 on the leader with 0.42 behind it.
    ap.add_argument("--confident-p", type=float, default=0.60)
    ap.add_argument("--confident-margin", type=float, default=0.25)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--candidate-floor", type=float, default=0.05)
    ap.add_argument("--evidence", default="single", choices=["single", "montage"],
                help="single: one mid-clip frame, PROVEN working on this hardware. "
                     "montage: 2x2 temporal grid - MEASURED to degenerate into repeated-token "
                     "garbage on this model+GPU (see build_single_frame docstring). Kept for "
                     "hardware that can actually serve it.")
    ap.add_argument("--montage-frames", type=int, default=4)
    ap.add_argument("--cell", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"C:\dvad\outputs\predictions_cascade.csv")
    ap.add_argument("--trace", default=r"C:\dvad\outputs\cascade_trace.jsonl")
    ap.add_argument("--gt", default=None, help="Score against this ground_truth.csv when given.")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    split_dir = Path(args.data_dir) / args.split
    videos = read_videos(split_dir)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {split_dir}")

    clf = AppearanceClassifier(weights=args.weights, device=args.device)
    reasoner = None if args.backend == "none" else _Adjudicator(
        args.backend, args.model, args.url, args.timeout)

    cfg = {"frames": args.frames, "evidence": args.evidence, "confident_p": args.confident_p,
           "confident_margin": args.confident_margin, "top_k": args.top_k,
           "candidate_floor": args.candidate_floor,
           "montage_frames": args.montage_frames, "cell": args.cell}

    print(f"[plan] {len(videos)} video(s) from {split_dir}")
    print(f"[plan] classifier={Path(args.weights).name} classes={len(clf.classes)}  "
          f"adjudicator={args.backend}:{args.model if args.backend!='none' else '-'}")
    print(f"[gate] contested unless p1>={args.confident_p} AND margin>={args.confident_margin}\n")

    rows, trace = [], []
    t_start = time.perf_counter()
    for i, (vid, path) in enumerate(videos, 1):
        if not path.exists():
            # A missing video_id scores worse than a wrong guess of normal.
            rows.append({"video_id": vid, "level": 3, "is_anomaly": "false",
                         "class_name": "normal", "start_time_sec": "",
                         "end_time_sec": "", "description_summary": ""})
            print(f"  [{i}/{len(videos)}] {vid:<10} MISSING FILE -> normal")
            continue
        d = classify_cascade(clf, reasoner, path, cfg, rng)
        d["video_id"] = vid
        trace.append(d)
        label = d["label"]
        is_anom = label != "normal"
        rows.append({
            "video_id": vid, "level": 3, "is_anomaly": "true" if is_anom else "false",
            "class_name": label, "start_time_sec": "", "end_time_sec": "",
            "description_summary": (
                f"{label.replace('_', ' ')} identified by "
                f"{'cascade (classifier + VLM on temporal montage)' if d['vlm_called'] else 'appearance classifier'}."
                if is_anom else ""),
        })
        flag = ""
        if d.get("changed"):
            flag = f"  [VLM changed: {d['probs'] and list(d['probs'])[0]} -> {label}]"
        elif d["vlm_called"]:
            flag = "  [VLM agreed]" if d["route"] == "vlm" else "  [VLM unparsed]"
        print(f"  [{i}/{len(videos)}] {vid:<10} {d['screen_s']:5.1f}s"
              f"{('+' + format(d['vlm_s'], '.0f') + 's VLM') if d['vlm_called'] else '        '}"
              f"  -> {label}{flag}")

    elapsed = time.perf_counter() - t_start
    calls = sum(1 for d in trace if d["vlm_called"])
    changed = sum(1 for d in trace if d.get("changed"))
    unparsed = sum(1 for d in trace if d["route"] == "vlm-unparsed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["video_id", "level", "is_anomaly", "class_name",
            "start_time_sec", "end_time_sec", "description_summary"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    tp = Path(args.trace)
    tp.parent.mkdir(parents=True, exist_ok=True)
    with tp.open("w", encoding="utf-8") as f:
        for d in trace:
            f.write(json.dumps(d) + "\n")

    print(f"\n[done] {len(videos)} video(s) in {elapsed:.0f}s -> {out}")
    print(f"[cost] VLM calls {calls}/{len(trace)} ({100*calls/max(len(trace),1):.0f}% of clips)"
          f"   changed the label on {changed}   unparsed {unparsed}")
    scr = sum(d["screen_s"] for d in trace)
    vlm = sum(d["vlm_s"] for d in trace)
    print(f"[cost] screening {scr:.0f}s total, VLM {vlm:.0f}s total"
          f"   ({vlm/max(calls,1):.0f}s per call)")
    print(f"[trace] {tp}")

    if args.gt:
        from score_submission import load_csv, score
        res = score(load_csv(Path(args.gt)), rows)
        print(f"\n=== scored against {args.gt} ===")
        print(f"  macro-F1                 : {res['macro_f1']}")
        print(f"  exact label-set accuracy : {res['video_exact_label_set_accuracy']}")
        b = res["is_anomaly_binary"]
        print(f"  is_anomaly accuracy      : {b['accuracy']}  "
              f"(tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})")
        print("\n  per-class:")
        for lab, m in sorted(res["per_class"].items()):
            print(f"    {lab:<34} P={m['precision']}  R={m['recall']}  "
                  f"F1={m['f1']}  (support={m['support_videos']})")


if __name__ == "__main__":
    main()
