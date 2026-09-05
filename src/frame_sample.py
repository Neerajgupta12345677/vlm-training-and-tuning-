"""Duration-aware, motion-aware frame picking.

Uniform linspace across a whole file is a coverage prior. It is the right
default on a 6-second Level-1 clip and the wrong default on a 6-minute Level-3
clip: it keeps twenty identical jam frames and misses a 1-second crash.

The 2026 VAD samplers (QVAD, Cerberus) do a cheap cascade instead:
  1. a uniform skeleton so the timeline is covered,
  2. motion peaks (frame-to-frame residual) so transients survive,
  3. first and last of the window as anchors.

YOLO peaks are optional and off by default - they need a GPU and we already
pay that at inference. Training can stay CPU-only.

    python src\\frame_sample.py --source C:\\dvad\\data\\ahc\\test\\videos\\T028.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import probe_video


# grab() through a long H.264 GOP is slower than one keyframe seek. Use
# sequential grab only when the next keep-frame is a few frames away.
_SEEK_GAP = 12


def _open_video(video: Path) -> tuple[cv2.VideoCapture | None, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, 25.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, n_total


def read_indices(video: Path, idxs: list[int],
                 cap: cv2.VideoCapture | None = None,
                 n_total: int | None = None) -> list[tuple[int, np.ndarray]]:
    """RGB frames at the given indices. Seeks across large gaps, grabs nearby."""
    own = cap is None
    if own:
        cap, _, n_total_got = _open_video(video)
        if cap is None:
            return []
        n_total = n_total_got
    assert cap is not None
    total = int(n_total or 0)
    want = sorted({int(np.clip(i, 0, max(total - 1, 0))) for i in idxs})
    out: list[tuple[int, np.ndarray]] = []
    next_pos = -1
    for i in want:
        if next_pos < 0 or i < next_pos or (i - next_pos) > _SEEK_GAP:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        else:
            for _ in range(i - next_pos):
                if not cap.grab():
                    break
        ok, fr = cap.read()
        next_pos = i + 1
        if ok and fr is not None:
            out.append((i, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    if own:
        cap.release()
    return out


def _motion_scores(frames: list[np.ndarray]) -> np.ndarray:
    """Mean absolute difference vs the previous frame. First frame scores 0."""
    if not frames:
        return np.zeros(0)
    scores = [0.0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev = cv2.GaussianBlur(prev, (5, 5), 0)
    for fr in frames[1:]:
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        scores.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    return np.asarray(scores, dtype=np.float32)


def pick_indices(n_total: int, n_keep: int, start: int = 0, end: int | None = None,
                 motion: np.ndarray | None = None) -> list[int]:
    """Mix uniform coverage with optional motion peaks inside [start, end)."""
    if n_total <= 0 or n_keep <= 0:
        return []
    end = n_total if end is None else min(end, n_total)
    start = max(0, min(start, end - 1))
    span = max(end - start, 1)
    n_keep = min(n_keep, span)
    # Always keep the window's ends so a short event at the edge is not dropped.
    uniform = np.linspace(start, end - 1, num=max(n_keep // 2, 2)).astype(int)
    chosen = set(int(x) for x in uniform.tolist())
    chosen.add(start)
    chosen.add(end - 1)
    if motion is not None and len(motion) == n_total:
        # Rank remaining slots by motion, but only inside the window.
        order = np.argsort(motion[start:end])[::-1]
        for off in order:
            if len(chosen) >= n_keep:
                break
            chosen.add(start + int(off))
    # Fill any leftover with extra uniform points.
    if len(chosen) < n_keep:
        extra = np.linspace(start, end - 1, num=n_keep).astype(int)
        for x in extra:
            chosen.add(int(x))
            if len(chosen) >= n_keep:
                break
    return sorted(chosen)[:n_keep]


def sample_window(video: Path, n: int, t0: float | None = None, t1: float | None = None,
                  candidates: int = 8, fps: float | None = None,
                  n_total: int | None = None) -> list[tuple[float, np.ndarray]]:
    """n RGB frames from [t0, t1] seconds (whole clip if either is None).

    Opens the file once. Decodes an 8-point candidate grid (seek, not a full
    walk), scores motion on that grid, and keeps a subset of those frames.
    """
    cap, fps_got, n_got = _open_video(video)
    if cap is None:
        return []
    if fps is None:
        fps = fps_got
    if n_total is None:
        n_total = n_got
    if n_total <= 0:
        cap.release()
        return []
    duration = n_total / fps
    t0 = 0.0 if t0 is None else max(0.0, t0)
    t1 = duration if t1 is None else min(duration, t1)
    if t1 <= t0:
        t1 = min(duration, t0 + 1.0 / fps)
    i0 = int(t0 * fps)
    i1 = max(int(np.ceil(t1 * fps)), i0 + 1)
    i1 = min(i1, n_total)
    span = i1 - i0
    # Tiny / low-fps clips: every frame. Else an 8-point grid is enough to
    # rank motion without walking a 4-minute file.
    n_cand = span if (fps < 5.0 or span <= max(n, 8)) else min(candidates, span)
    n_cand = max(n_cand, min(n, span))
    cand_idx = np.linspace(i0, i1 - 1, num=n_cand).astype(int)
    grabbed = read_indices(video, cand_idx.tolist(), cap=cap, n_total=n_total)
    cap.release()
    if not grabbed:
        return []
    frames = [f for _, f in grabbed]
    scores = _motion_scores(frames)
    local_keep = pick_indices(len(grabbed), min(n, len(grabbed)),
                              start=0, end=len(grabbed), motion=scores)
    return [(grabbed[j][0] / fps, grabbed[j][1]) for j in local_keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    args = ap.parse_args()
    got = sample_window(Path(args.source), args.n, args.t0, args.t1)
    meta = probe_video(args.source)
    print(f"{args.source}  {meta.frame_count}fr @{meta.fps:.1f}fps  kept {len(got)}")
    for t, fr in got:
        print(f"  t={t:7.2f}s  {fr.shape[1]}x{fr.shape[0]}")


if __name__ == "__main__":
    main()
