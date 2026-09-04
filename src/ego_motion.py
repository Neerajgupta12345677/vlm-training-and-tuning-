"""Camera-motion compensation. Without this, none of Stage 2 works on a drone.

Every Stage 2 signal - dwell time, speed, the stop anchor, zone membership - is
computed from where an object sits in the IMAGE. That is only the same thing as
where it sits in the WORLD when the camera is bolted down.

A drone pans, drifts while hovering, and orbits. On that footage a parked car
has non-zero image velocity, so `stationary_since_s` never latches and the
flagship stopped-vehicle rule silently reports nothing. Nothing errors; the
system just goes quiet. Fixed-pixel zone polygons rot the same way: one
translation and every polygon points at the wrong piece of road.

The fix is to estimate the frame-to-frame background transform and express
track positions in a stabilised reference frame (that of the first frame).
Stage 2's arithmetic then works unchanged, because in that frame a parked car
really is stationary.

Approach: shi-tomasi corners + pyramidal Lucas-Kanade flow, then a partial
affine (translation, rotation, uniform scale) fitted with RANSAC. Partial affine
rather than full homography because it has 4 degrees of freedom instead of 8 -
far more stable to fit from sparse noisy correspondences, and drone motion over
a short window is well described by pan/rotate/zoom.

Two deliberate refusals, in keeping with the rest of the system: if too few
correspondences survive, it reports failure instead of returning a bad
transform; and it reports how much the camera is actually moving, so callers can
say whether compensation was needed at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class EgoMotion:
    """Result for one frame."""

    ok: bool                  # was a transform estimated at all
    matrix: np.ndarray | None  # 2x3 affine mapping PREVIOUS frame -> current
    inliers: int
    translation_px: float      # magnitude of this frame's camera shift
    static_camera: bool        # camera effectively bolted down


@dataclass
class EgoMotionEstimator:
    """Tracks cumulative camera motion and maps points into a stable frame."""

    max_corners: int = 600
    quality: float = 0.01
    min_distance: int = 12
    work_width: int = 640      # estimate on a downscaled copy; it is much cheaper
    min_inliers: int = 12
    static_threshold_px: float = 1.0   # per-frame shift below this = static rig
    redetect_every: int = 12           # refresh corners periodically

    _prev_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_pts: np.ndarray | None = field(default=None, init=False, repr=False)
    _scale: float = field(default=1.0, init=False)
    _frames: int = field(default=0, init=False)
    # Cumulative transform: reference (first) frame -> current frame, 3x3.
    _cum: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64), init=False)
    _last: EgoMotion | None = field(default=None, init=False, repr=False)
    _fail_count: int = field(default=0, init=False)
    _shift_history: list = field(default_factory=list, init=False, repr=False)

    # -- internals ---------------------------------------------------------
    def _prep(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = self.work_width / float(w) if w > self.work_width else 1.0
        if self._scale < 1.0:
            frame = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _detect(self, gray: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                      qualityLevel=self.quality,
                                      minDistance=self.min_distance)
        return pts if pts is not None and len(pts) >= self.min_inliers else None

    # -- public ------------------------------------------------------------
    def update(self, frame: np.ndarray) -> EgoMotion:
        """Ingest a frame, returning this frame's camera motion."""
        gray = self._prep(frame)
        self._frames += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray)
            self._last = EgoMotion(True, np.array([[1, 0, 0], [0, 1, 0]], np.float64),
                                   0, 0.0, True)
            return self._last

        if self._prev_pts is None or len(self._prev_pts) < self.min_inliers:
            self._prev_pts = self._detect(self._prev_gray)

        result = EgoMotion(False, None, 0, 0.0, False)
        if self._prev_pts is not None:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts.astype(np.float32), None)
            if nxt is not None and status is not None:
                keep = status.reshape(-1).astype(bool)
                src = self._prev_pts.reshape(-1, 2)[keep]
                dst = nxt.reshape(-1, 2)[keep]
                if len(src) >= self.min_inliers:
                    M, inl = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                        maxIters=2000, confidence=0.99)
                    n_inl = int(inl.sum()) if inl is not None else 0
                    if M is not None and n_inl >= self.min_inliers:
                        # Rescale translation back to full-resolution pixels.
                        M_full = M.copy()
                        if self._scale != 1.0:
                            M_full[:, 2] /= self._scale
                        shift = float(np.hypot(M_full[0, 2], M_full[1, 2]))
                        result = EgoMotion(True, M_full, n_inl, shift,
                                           shift < self.static_threshold_px)
                        H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
                        self._cum = H @ self._cum
                        self._prev_pts = dst.reshape(-1, 1, 2)

        if not result.ok:
            self._fail_count += 1
        if self._frames % self.redetect_every == 0 or not result.ok:
            self._prev_pts = self._detect(gray)
        self._prev_gray = gray
        self._shift_history.append(result.translation_px if result.ok else 0.0)
        self._last = result
        return result

    def to_reference(self, x: float, y: float) -> tuple[float, float]:
        """Map a CURRENT-frame point into the stabilised reference frame.

        Stage 2 stores history in this space, so a parked car keeps a constant
        position even while the drone moves.
        """
        try:
            inv = np.linalg.inv(self._cum)
        except np.linalg.LinAlgError:
            return x, y
        # Reference frame works in the downscaled space the transform was fitted
        # in, so scale in and back out to keep units consistent.
        s = self._scale if self._scale else 1.0
        p = inv @ np.array([x * s, y * s, 1.0])
        w = p[2] if abs(p[2]) > 1e-9 else 1.0
        return float(p[0] / w / s), float(p[1] / w / s)

    @property
    def camera_is_moving(self) -> bool:
        """True if the recent camera shift is beyond static-rig noise."""
        recent = self._shift_history[-30:]
        if not recent:
            return False
        return float(np.median(recent)) >= self.static_threshold_px

    def stats(self) -> dict:
        recent = self._shift_history[-120:] or [0.0]
        return {
            "frames": self._frames,
            "estimation_failures": self._fail_count,
            "median_shift_px": round(float(np.median(recent)), 2),
            "max_shift_px": round(float(np.max(recent)), 2),
            "camera_is_moving": self.camera_is_moving,
            "last_inliers": self._last.inliers if self._last else 0,
        }
