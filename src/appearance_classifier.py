"""Stage 1.5: appearance classifier for conditions a tracker cannot see.

Stage 2 reasons about how objects MOVE. Fire, smoke, flooding and road debris
are things a frame simply CONTAINS, so no dwell/speed/zone rule can represent
them - measured as F1 0.0 on all four classes across the organisers' public
test set, on 9 videos of support, because `--decision rules` never calls a
model and no rule can fire.

This is the cheap always-available answer to that. MobileNetV3-Small, 2.54M
parameters, ~5MB on disk, batch inference over a handful of sampled frames.
It runs in well under a second per clip on the GTX 1650, against 27-45s for a
single qwen2.5vl:3b call, so it stays inside the "small model, real time,
economical across many feeds" constraint the brief is built around.

Two output modes:
  classify_video()   - one verdict for the whole clip (Level 1 short clips)
  classify_windows() - verdicts over time, merged into intervals (Level 2/3)

    python src\appearance_classifier.py --source <video.mp4>
    python src\appearance_classifier.py --data_dir C:\dvad\data\ahc --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from common import MODELS_DIR

DEFAULT_WEIGHTS = Path(MODELS_DIR) / "appearance_classifier.pt"

# Fallback allowlist for checkpoints that predate the 11-class model. Live
# behaviour derives the label set from the checkpoint itself (see __init__):
# hardcoding it here meant retraining with new classes silently produced a model
# whose extra outputs could never be asserted.
#
# "normal" is a real output of the network but means "nothing visible here, let
# Stage 2 decide" - it never becomes a submission row on its own.
APPEARANCE_LABELS = {
    "fire", "smoke", "waterlogging_or_flood", "road_spill_or_debris",
    "fighting_or_violence", "traffic_accident",
}


class AppearanceClassifier:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cuda",
                 threshold: float = 0.72) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found - train it first:\n"
                f"  python src\\train_appearance.py --data_dir C:\\dvad\\data\\ahc"
            )
        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.img_size: int = ckpt.get("img_size", 224)
        self.threshold = threshold
        # Anything the network was trained to emit except `normal` is assertable.
        self.labels: set[str] = {c for c in self.classes if c != "normal"}

        model = mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # -- core ---------------------------------------------------------------
    def _probs(self, frames: list[np.ndarray]) -> np.ndarray:
        """(N, C) softmax probabilities for a list of RGB frames."""
        if not frames:
            return np.zeros((0, len(self.classes)), np.float32)
        batch = torch.stack([self.tf(f) for f in frames]).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(batch), dim=1).cpu().numpy()

    def _read(self, video: Path, n: int) -> tuple[list[np.ndarray], list[float]]:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: list[np.ndarray] = []
        times: list[float] = []
        if total <= 0:
            cap.release()
            return frames, times
        for i in np.linspace(total * 0.05, total * 0.95, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(i, 0, total - 1)))
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                times.append(float(i) / fps)
        cap.release()
        return frames, times

    def classify_video(self, video: str | Path, n_frames: int = 8) -> dict | None:
        """One verdict for the whole clip, or None if nothing appearance-based.

        Averages probabilities across frames before deciding, rather than
        voting per frame: smoke and flood drift in and out of view, and the
        mean is far steadier than any single frame's argmax.
        """
        frames, _ = self._read(Path(video), n_frames)
        probs = self._probs(frames)
        if probs.size == 0:
            return None
        mean = probs.mean(axis=0)
        idx = int(mean.argmax())
        label, conf = self.classes[idx], float(mean[idx])
        if label not in self.labels or conf < self.threshold:
            return None
        return {"class_name": label, "confidence": round(conf, 4),
                "scores": {c: round(float(p), 4) for c, p in zip(self.classes, mean)}}

    def score_video(self, video: str | Path, n_frames: int = 8) -> dict[str, float] | None:
        """Clip-mean probabilities over every class, with no threshold applied.

        Kept separate from classify_video so a threshold sweep does not have to
        re-run inference per candidate value. Dumping these once for a split
        turns threshold selection into an arithmetic problem over a small JSON
        file rather than a repeated GPU pass over every video.
        """
        frames, _ = self._read(Path(video), n_frames)
        probs = self._probs(frames)
        if probs.size == 0:
            return None
        mean = probs.mean(axis=0)
        return {c: round(float(p), 5) for c, p in zip(self.classes, mean)}

    def classify_windows(self, video: str | Path, n_frames: int = 24,
                         merge_gap_s: float = 6.0, min_span_s: float = 1.0) -> list[dict]:
        """Timed intervals for appearance classes, for Level 2/3 localisation.

        Samples across the clip, keeps frames whose top class is an appearance
        hazard above threshold, then merges neighbouring hits of the same label
        into one interval. Without this, a 10-minute video could only ever get
        an untimed whole-clip label and would score 0 temporal IoU.
        """
        frames, times = self._read(Path(video), n_frames)
        probs = self._probs(frames)
        if probs.size == 0:
            return []
        hits: list[tuple[float, str, float]] = []
        for t, row in zip(times, probs):
            idx = int(row.argmax())
            label, conf = self.classes[idx], float(row[idx])
            if label in self.labels and conf >= self.threshold:
                hits.append((t, label, conf))
        if not hits:
            return []

        out: list[dict] = []
        for label in sorted({h[1] for h in hits}):
            spans = [h for h in hits if h[1] == label]
            start = prev = spans[0][0]
            confs = [spans[0][2]]
            for t, _, c in spans[1:]:
                if t - prev <= merge_gap_s:
                    prev = t
                    confs.append(c)
                    continue
                out.append({"class_name": label, "start_time_sec": round(start, 2),
                            "end_time_sec": round(max(prev, start + min_span_s), 2),
                            "confidence": round(sum(confs) / len(confs), 4)})
                start = prev = t
                confs = [c]
            out.append({"class_name": label, "start_time_sec": round(start, 2),
                        "end_time_sec": round(max(prev, start + min_span_s), 2),
                        "confidence": round(sum(confs) / len(confs), 4)})
        return sorted(out, key=lambda r: r["start_time_sec"])


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1.5 appearance classifier.")
    p.add_argument("--source", help="A single video.")
    p.add_argument("--data_dir", default=None, help="AHC dataset root, to sweep a split.")
    p.add_argument("--split", default="test")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--windows", action="store_true", help="Report timed intervals too.")
    p.add_argument("--dump", default=None,
                   help="Write per-video class probabilities to this JSON, for threshold sweeps.")
    args = p.parse_args()

    clf = AppearanceClassifier(args.weights, threshold=args.threshold)
    print(f"[ok] classes={clf.classes} threshold={args.threshold}")

    videos: list[Path] = []
    if args.source:
        videos = [Path(args.source)]
    elif args.data_dir:
        vdir = Path(args.data_dir) / args.split / "videos"
        videos = sorted(vdir.glob("*.mp4"))
    else:
        raise SystemExit("Pass --source or --data_dir")

    dumped: dict[str, dict[str, float]] = {}
    for v in videos:
        if args.dump:
            scores = clf.score_video(v, n_frames=args.frames)
            if scores is not None:
                dumped[v.stem] = scores
                top = max(scores, key=scores.get)
                print(f"  {v.stem:<10} {top:<34} {scores[top]:.3f}")
            else:
                print(f"  {v.stem:<10} unreadable")
            continue
        res = clf.classify_video(v, n_frames=args.frames)
        verdict = f"{res['class_name']} ({res['confidence']:.3f})" if res else "-"
        print(f"  {v.stem:<10} {verdict}")
        if args.windows and res:
            for w in clf.classify_windows(v):
                print(f"      {w['start_time_sec']:>7.1f}-{w['end_time_sec']:<7.1f} "
                      f"{w['class_name']} ({w['confidence']:.2f})")

    if args.dump:
        import json

        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(
            {"classes": clf.classes, "frames": args.frames, "scores": dumped}, indent=2))
        print(f"\n[dump] {len(dumped)} video(s) -> {args.dump}")


if __name__ == "__main__":
    main()
