"""Inference for the MOTION classifier - the motion/duration classes.

`AppearanceClassifier` reads RGB frames; this reads ego-compensated temporal
DIFFERENCES at the same sample times. Everything downstream of that - clip-mean
probabilities, thresholding, `classify_windows`, and the adaptive-rate
`windows_for_label` used for Level 2/3 timing - is identical, so this subclasses
it and overrides exactly one method: `_read`.

That is not a shortcut, it is the point. The window geometry, the padding that
fixed the T028 near-misses, and the centre-anchored span clamping were all
tuned against real failures; forking them for a second model would mean fixing
the next such bug twice.

Why a different input at all
----------------------------
The appearance model was given four motion classes and failed all of them on
the public leaderboard - loitering 2/7, vehicle_blocking 0/2, wrong_way 0/1,
fighting 0/3, together 10 of the 19 false alarms - while no appearance class
failed. Training clips are trimmed to the event (median 100% event coverage;
loitering has 810 in-event frames and 0 frames of the same scene idle), so
class correlates perfectly with background, and background is constant within a
video. A score that does not vary across a clip cannot localise, which is why
Level 3 scored 20%.

Differencing removes the background, so the network is pushed onto motion,
which both defines these classes and varies within a clip.

Cost note: each sample decodes 4 frames (current + three lags) and fits a
RANSAC partial affine per lag, so this is meaningfully slower per sample than
reading one RGB frame. It runs on sampled frames only - never in the per-frame
Stage 1 loop - so it does not affect the real-time path.

    python src\\motion_classifier.py --video C:\\dvad\\data\\ahc\\test\\videos\\T034.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from appearance_classifier import AppearanceClassifier
from build_motion_frames import LAGS_S, motion_image
from common import MODELS_DIR

DEFAULT_WEIGHTS = Path(MODELS_DIR) / "motion_classifier.pt"

# Classes this model owns. The appearance model keeps fire/smoke/waterlogging/
# road_spill, which genuinely are visible in one frame.
MOTION_LABELS = {
    "loitering_or_suspicious_presence",
    "wrong_way_driving",
    "traffic_congestion",
    "vehicle_blocking_traffic",
    "fighting_or_violence",
    "traffic_accident",
}


class MotionClassifier(AppearanceClassifier):
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cuda",
                 threshold: float = 0.72) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found - build the cache and train it first:\n"
                f"  python src\\build_motion_frames.py --data_dir C:\\dvad\\data\\ahc\n"
                f"  python src\\train_motion.py --cache C:\\dvad\\data\\motion_frames"
            )
        super().__init__(weights, device=device, threshold=threshold)

    def _read(self, video: Path, n: int) -> tuple[list[np.ndarray], list[float]]:
        """Motion-difference images at n evenly spaced times.

        Samples start at `max(LAGS_S)` rather than at 5% of the clip: before
        that point the longest baseline would be clamped to t=0, silently
        shortening it and making early samples look quieter than they are.

        A sample whose ego-motion fit fails is DROPPED, not substituted - the
        parallel `times` list keeps the surviving samples aligned to their real
        timestamps, so a dropped frame costs a little temporal resolution and
        never shifts a window.
        """
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: list[np.ndarray] = []
        times: list[float] = []
        if total <= 0 or fps <= 0:
            cap.release()
            return frames, times

        duration = total / fps
        start = min(max(LAGS_S), duration * 0.05)
        for t in np.linspace(start, duration * 0.95, max(n, 1)):
            img = motion_image(cap, float(t), fps, total, self.img_size)
            if img is None:
                continue
            # motion_image returns BGR-ordered channels only in the sense that
            # cv2 wrote them; the trained cache was read back through the same
            # cv2.imread path, so converting to RGB here reproduces training.
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            times.append(float(t))
        cap.release()
        return frames, times


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=r"C:\dvad\data\ahc",
                   help="Kept so every script in src/ takes the same flag.")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--n-frames", type=int, default=24)
    args = p.parse_args()

    clf = MotionClassifier(args.weights, threshold=args.threshold)
    print(f"[motion] classes={clf.classes} @ {args.threshold}")
    verdict = clf.classify_video(Path(args.video), n_frames=args.n_frames)
    print(f"[verdict] {verdict}")
    scores = clf.scores_for_video(Path(args.video), n_frames=args.n_frames) \
        if hasattr(clf, "scores_for_video") else None
    if scores:
        for c, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<34} {s:.3f}")


if __name__ == "__main__":
    main()
