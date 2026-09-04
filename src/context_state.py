"""STAGE 2 - every frame, zero model calls: per-track state and event triggers.

This stage is what keeps the pipeline affordable: it decides which ~1-5% of
frames deserve a Stage 3 VLM call. Everything here is arithmetic.

Two design choices worth knowing:

1. Speed is scale-invariant. Raw pixel speed is meaningless in drone footage
   because altitude changes the pixels-per-metre ratio constantly. We divide
   by the object's own box diagonal, giving "body-lengths per second", so the
   stationary test holds whether the drone is at 20m or 120m.

2. We track the stopped-neighbour ratio. One car stopped in a flowing lane is
   an anomaly; every car stopped is a traffic jam. Same per-track state, very
   different verdict - so we hand that ratio to the VLM as context.

Self-test on synthetic tracks (no video, no model needed):
    python src\\context_state.py --selftest
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
PERSON_NAMES = {"person"}

# Context strings go into the VLM prompt *and* into the training targets, so
# they have to read as clean English. Naive formatting produced "in a unknown".
ZONE_PHRASE = {
    "driving_lane": "a live driving lane",
    "shoulder": "the hard shoulder",
    "parking": "a parking area",
    "sidewalk": "a sidewalk",
    "restricted": "a restricted area",
    "unknown": "an unmapped part of the scene",
}


def phrase_zone(kind: str) -> str:
    return ZONE_PHRASE.get(kind, f"a {kind.replace('_', ' ')}")


# Scale calibration with no training and no camera intrinsics: a detected
# vehicle is its own ruler. From above, a box's long side is roughly the
# vehicle's length, so metres-per-pixel falls out of the class alone.
#
# Deliberately per-track rather than per-scene: a vehicle far up the road has
# both a smaller box AND smaller pixel motion, so calibrating each track from
# its own box cancels perspective instead of fighting it.
EXPECTED_LENGTH_M = {
    "car": 4.4,
    "truck": 8.0,
    "bus": 11.0,
    "motorcycle": 2.1,
    "bicycle": 1.8,
    "person": 0.6,      # footprint from above, not height
}


def metres_per_pixel(class_name: str, xyxy) -> float | None:
    """Scene scale implied by one detection, or None for unknown classes."""
    length_m = EXPECTED_LENGTH_M.get(class_name)
    if not length_m:
        return None
    x1, y1, x2, y2 = xyxy
    long_px = max(abs(x2 - x1), abs(y2 - y1))
    return length_m / long_px if long_px > 2 else None


# --- Zones -----------------------------------------------------------------
@dataclass
class Zone:
    name: str
    kind: str  # driving_lane | shoulder | parking | sidewalk | restricted
    polygon: np.ndarray  # (N, 2) int32
    flow_deg: float | None = None  # expected travel direction, for wrong-way checks

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0


class ZoneMap:
    """Zones come from a one-time per-video calibration, never per-frame inference."""

    def __init__(self, zones: list[Zone], default_kind: str = "unknown") -> None:
        self.zones = zones
        self.default_kind = default_kind

    @classmethod
    def load(cls, path: str | Path | None, default_kind: str = "unknown") -> "ZoneMap":
        if not path:
            return cls([], default_kind=default_kind)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = [
            Zone(
                name=z["name"],
                kind=z["kind"],
                polygon=np.array(z["polygon"], dtype=np.int32),
                flow_deg=z.get("flow_deg"),
            )
            for z in data["zones"]
        ]
        return cls(zones, default_kind=data.get("default_kind", default_kind))

    def lookup(self, x: float, y: float) -> tuple[str, Zone | None]:
        for z in self.zones:
            if z.contains(x, y):
                return z.kind, z
        return self.default_kind, None


# --- Per-track state -------------------------------------------------------
@dataclass
class TrackState:
    track_id: int
    class_name: str
    first_seen_s: float
    last_seen_s: float
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, cx, cy, diag)
    stationary_since_s: float | None = None
    slow_since_s: float | None = None  # first frame it looked slow (for reporting)
    # Rolling record of "was this frame slow relative to ambient?". Consistency
    # over these is the trigger, not elapsed time - see TriggerConfig.
    _slow_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    last_slow_ratio: float = 1.0
    stop_anchor: tuple[float, float] | None = None  # where it was when it stopped
    frame_diag: float = 1.0   # frame diagonal at ingest; used by the min-size gate
    box_diag: float = 0.0     # this object's box diagonal, same units
    zone_kind: str = "unknown"
    zone_name: str | None = None
    last_xyxy: tuple[float, float, float, float] = (0, 0, 0, 0)
    norm_speed: float = 0.0
    scale_rate: float = 0.0          # instantaneous |d(box size)|/size/sec - very noisy
    scale_rate_med: float = 0.0      # median over the recent window - the usable signal
    _scale_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    px_speed: float = 0.0            # pixels/sec, before scale conversion
    speed_kmh: float | None = None   # real-world estimate via the car-length ruler
    m_per_px: float | None = None
    heading_deg: float | None = None
    vlm_calls: int = 0
    last_trigger_s: float | None = None

    @property
    def age_s(self) -> float:
        return self.last_seen_s - self.first_seen_s

    @property
    def stationary_s(self) -> float:
        if self.stationary_since_s is None:
            return 0.0
        return max(0.0, self.last_seen_s - self.stationary_since_s)

    @property
    def slow_s(self) -> float:
        if self.slow_since_s is None:
            return 0.0
        return max(0.0, self.last_seen_s - self.slow_since_s)

    @property
    def slow_fraction(self) -> float:
        """Share of recent observations where this vehicle was crawling."""
        if not self._slow_hist:
            return 0.0
        return sum(self._slow_hist) / len(self._slow_hist)

    @property
    def slow_observations(self) -> int:
        return len(self._slow_hist)

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in VEHICLE_NAMES

    @property
    def is_person(self) -> bool:
        return self.class_name in PERSON_NAMES


@dataclass
class Event:
    """A Stage 2 trigger. The only thing that may cause a Stage 3 VLM call."""

    frame_idx: int
    timestamp_s: float
    kind: str
    track_id: int | None
    class_name: str
    zone_kind: str
    bbox: tuple[float, float, float, float]
    context: str
    features: dict
    # Stage 2's own verdict, computed from measured facts alone. This is the
    # reliable half of the decision: dwell time, zone and neighbour state are
    # arithmetic, not perception, so no model can improve on them.
    rule_anomalous: bool = False
    rule_severity: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "kind": self.kind,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "zone_kind": self.zone_kind,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "context": self.context,
            "features": self.features,
            "rule_anomalous": self.rule_anomalous,
            "rule_severity": round(self.rule_severity, 2),
        }
        return d


@dataclass
class TriggerConfig:
    """Thresholds. Tune stop_seconds down for short test clips."""

    stop_seconds: float = 20.0
    stationary_speed: float = 0.05  # body-lengths/sec below which we call it stopped
    person_in_lane_seconds: float = 2.0
    loiter_seconds: float = 25.0    # a person stationary this long is loitering
    # "Abnormally slow" is deliberately RELATIVE to surrounding traffic, never an
    # absolute km/h. That makes congestion self-cancelling: in a jam the ambient
    # median drops too, so a crawling vehicle is no longer an outlier and the
    # rule stays quiet. Added after the 27B teacher flagged two cases our rules
    # missed - both "truck moving at only 6 km/h in a live lane".
    # OFF BY DEFAULT - deliberate. The rule is real and works, but it was tuned
    # against a single oblique clip, and every threshold below was set by
    # chasing false positives on that one video. On unknown footage that is a
    # liability, and it costs the zero-false-positive property that is the
    # strongest claim this system has. Enable with --enable-slow-vehicle once
    # you have footage you can actually validate it against.
    enable_slow_vehicle: bool = False
    slow_ratio: float = 0.35        # below this fraction of ambient speed = crawling
    slow_min_neighbours: int = 3    # need this many moving vehicles for a valid ambient
    # Judged on CONSISTENCY, not elapsed time. Measured on this footage, tracks
    # live only ~0.7-0.8s (vehicles cross frame fast, and ByteTrack re-acquires),
    # so any multi-second dwell requirement can never be satisfied and the rule
    # would be dead code. Instead: enough observations to be worth judging, and
    # a clear majority of them slow.
    slow_min_observations: int = 6  # fewer than this is not evidence, it is noise
    slow_fraction: float = 0.7      # this share of recent observations must be slow
    # Deliberately stricter than min_box_diag_frac (0.02) used by the stop rule.
    # "Stopped" is a blunt dwell measurement that survives a small noisy box;
    # "abnormally slow" is a RATIO of two small, noisy quantities, so it needs a
    # clearer view of the object before the answer means anything. Measured:
    # 72x52 and 83x98 boxes far up the road were the only remaining aerial
    # false positives, and both sit below this floor.
    slow_min_box_diag_frac: float = 0.05
    crowd_count: int = 8            # live person tracks that constitute a crowd
    crowd_cooldown_s: float = 45.0
    wrong_way_tolerance_deg: float = 100.0
    wrong_way_min_speed: float = 0.3
    cooldown_seconds: float = 30.0
    max_calls_per_track: int = 3
    speed_window_s: float = 1.0
    min_track_age_s: float = 1.0  # ignore brand-new tracks; ByteTrack ids are unstable early
    # A stopped vehicle only counts as having moved off once its centre travels
    # this many body-lengths from where it stopped. Instantaneous speed alone is
    # too fragile: box jitter on a parked truck repeatedly cleared the dwell
    # latch and reset a genuine 19s stop to 2s.
    move_release_bodylengths: float = 0.6
    # Max scale spread (widest/narrowest metres-per-pixel) for a km/h estimate
    # to be trustworthy. ~2x still reads as near-nadir; well beyond that the
    # view is oblique and along-road motion is compressed in the image plane.
    max_obliquity_for_speed: float = 2.5
    # Suppress a new event whose box overlaps one that already fired recently.
    # Agnostic NMS removes most duplicate detections, but a track can still be
    # split (id switch, occlusion), and two ids on one object means two alerts
    # for one incident - which reads as a broken system.
    duplicate_iou: float = 0.55
    duplicate_window_s: float = 20.0
    # Minimum object size, as a fraction of the frame diagonal, before we are
    # willing to claim it is stationary. A distant vehicle receding up the road
    # has almost no image-plane displacement, so it reads as stopped no matter
    # how fast it is really going. Raising detector recall on small objects
    # (the --aerial preset) made this a live false-positive source, so the rule
    # now declines to judge what it cannot measure - same principle as the
    # obliquity gate on speed.
    min_box_diag_frac: float = 0.02
    # Median apparent-size change per second above which an object counts as
    # moving in depth, even with a static centroid. Set from measurement:
    # parked truck median 0.009, approaching truck median 0.101. 0.04 sits
    # between them with margin on both sides.
    scale_rate_moving: float = 0.04


class ContextStateTracker:
    def __init__(self, zones: ZoneMap | None = None, config: TriggerConfig | None = None) -> None:
        self.zones = zones or ZoneMap([])
        self.cfg = config or TriggerConfig()
        self.tracks: dict[int, TrackState] = {}
        self.frames_seen = 0
        self.events_fired = 0
        self._last_crowd_s: float | None = None
        self._now = 0.0
        self._recent_alerts: list[tuple[float, tuple[float, float, float, float], str]] = []

    # -- helpers ------------------------------------------------------------
    def _update_kinematics(self, st: TrackState) -> None:
        """Compute scale-invariant speed + heading over the recent window."""
        if len(st.history) < 2:
            st.norm_speed = 0.0
            return
        t_now, cx_now, cy_now, diag_now = st.history[-1]
        cutoff = t_now - self.cfg.speed_window_s
        ref = st.history[0]
        for entry in st.history:
            if entry[0] >= cutoff:
                ref = entry
                break
        t_ref, cx_ref, cy_ref, _ = ref
        dt = t_now - t_ref
        if dt <= 1e-6:
            st.norm_speed = 0.0
            return
        dx, dy = cx_now - cx_ref, cy_now - cy_ref
        dist = float(np.hypot(dx, dy))
        scale = max(diag_now, 1e-6)
        st.norm_speed = (dist / scale) / dt
        st.px_speed = dist / dt
        # Motion along the viewing axis produces almost no centroid displacement
        # but a steady change in apparent size. Measured case: a DAF truck
        # driving straight at the camera was flagged "stationary 5s in a live
        # lane" because it barely moved in the image plane. Growth/shrink of the
        # box catches exactly that.
        diag_ref = ref[3]
        st.scale_rate = abs(diag_now - diag_ref) / max(diag_ref, 1e-6) / dt
        # Instantaneous scale rate is unusable on its own - measured on this
        # footage, a PARKED truck's box jitter reached 0.499/s and its p90
        # (0.207) exceeded an APPROACHING truck's median (0.101), so the two
        # populations overlap. The medians separate 11x (0.009 vs 0.101), so
        # gate on the median of the window instead.
        st._scale_hist.append(st.scale_rate)
        st.scale_rate_med = float(np.median(st._scale_hist))
        st.m_per_px = metres_per_pixel(st.class_name, st.last_xyxy)
        st.speed_kmh = st.px_speed * st.m_per_px * 3.6 if st.m_per_px else None
        if dist > 0.5 * scale:  # heading is noise unless it actually moved
            st.heading_deg = float(np.degrees(np.arctan2(dy, dx))) % 360.0

    def view_obliquity(self) -> float | None:
        """How far from top-down this camera is, from scale spread alone.

        In a nadir view every vehicle sits at roughly the same scale, so
        metres-per-pixel is near-constant across the frame. In an oblique view
        it varies strongly with depth. The ratio between the widest and
        narrowest scale is therefore a free obliquity estimate - no intrinsics,
        no horizon fitting.

        This matters because the car-length ruler only yields honest speeds in a
        near-nadir view: when vehicles move along the depth axis, their
        image-plane displacement badly understates real speed. Measured on a
        bridge-style view it reported 12 km/h for highway traffic. We would
        rather withhold the number than state a wrong one.
        """
        scales = [st.m_per_px for st in self.tracks.values()
                  if st.m_per_px and st.last_seen_s >= self._now - 2.0]
        if len(scales) < 3:
            return None
        lo, hi = min(scales), max(scales)
        return hi / lo if lo > 0 else None

    def _traffic_speed_kmh(self, exclude_id: int | None = None) -> tuple[float | None, bool]:
        """(median speed of moving vehicles, is_reliable).

        Unreliable means the geometry does not support the estimate; callers
        must not present the number as fact.
        """
        speeds = [
            st.speed_kmh for tid, st in self.tracks.items()
            if tid != exclude_id and st.is_vehicle and st.speed_kmh
            and st.last_seen_s >= self._now - 1.0
            and st.norm_speed >= self.cfg.stationary_speed
        ]
        if not speeds:
            return None, False
        obliquity = self.view_obliquity()
        reliable = obliquity is not None and obliquity <= self.cfg.max_obliquity_for_speed
        return float(np.median(speeds)), reliable

    @staticmethod
    def _total_motion(st: "TrackState") -> float:
        """Apparent motion combining the image plane AND depth.

        Neither component alone survives both camera geometries:
          * A near-nadir view puts nearly all motion in the image plane, so
            centroid speed works and scale change is ~0.
          * An oblique view (this highway footage) puts nearly all motion along
            the view axis, so centroid speed collapses toward 0 for everyone and
            an image-plane-only comparison is meaningless.
        Summing them gives one number that means "how much is this thing
        actually moving" in either geometry, which is what a speed comparison
        needs to be built on.
        """
        return st.norm_speed + st.scale_rate_med

    def _ambient_speed(self, exclude_id: int | None = None) -> tuple[float | None, int]:
        """Median total motion of MOVING vehicles in view, plus how many.

        Returns (None, n) when there are too few moving vehicles to establish a
        meaningful ambient - with nothing to compare against, "slow" is not a
        judgement we are entitled to make.
        """
        speeds = [
            self._total_motion(st) for tid, st in self.tracks.items()
            if tid != exclude_id and st.is_vehicle
            and st.last_seen_s >= self._now - 1.0
            and st.norm_speed >= self.cfg.stationary_speed
        ]
        if len(speeds) < self.cfg.slow_min_neighbours:
            return None, len(speeds)
        return float(np.median(speeds)), len(speeds)

    def _update_slow_latch(self, live_ids: set[int], det) -> None:
        """Latch how long each vehicle has been crawling relative to ambient.

        Runs after all tracks are ingested for the frame, because the ambient
        median depends on every track's updated speed. Latched rather than
        instantaneous so a car braking briefly for a junction does not fire.
        """
        for tid in live_ids:
            st = self.tracks[tid]
            if not st.is_vehicle:
                continue
            ambient, _n = self._ambient_speed(exclude_id=tid)
            if ambient is None:
                continue  # no ambient to compare against; record nothing
            moving = st.norm_speed >= self.cfg.stationary_speed
            if not moving:
                continue  # a full stop is Rule 1's business, not this one
            # Compare TOTAL motion (image plane + depth) against the same
            # measure for its neighbours. An earlier attempt gated on absolute
            # depth motion instead; that killed the aerial false positives but
            # also made the rule inert on oblique footage, where essentially
            # every vehicle is moving along the view axis. Comparing like with
            # like fixes both cases.
            ratio = self._total_motion(st) / ambient
            st.last_slow_ratio = ratio
            crawling = ratio < self.cfg.slow_ratio
            st._slow_hist.append(crawling)
            if crawling and st.slow_since_s is None:
                st.slow_since_s = det.timestamp_s
            elif not crawling and st.slow_fraction < self.cfg.slow_fraction:
                st.slow_since_s = None

    def _stopped_ratio(self, exclude_id: int | None = None) -> tuple[int, int]:
        """(stopped_vehicles, total_vehicles) among currently-live vehicle tracks."""
        stopped = total = 0
        for tid, st in self.tracks.items():
            if tid == exclude_id or not st.is_vehicle:
                continue
            if st.last_seen_s < self._now - 1.0:  # only count currently-visible
                continue
            total += 1
            if st.norm_speed < self.cfg.stationary_speed:
                stopped += 1
        return stopped, total

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _kind_family(kind: str) -> str:
        """Group rules that describe the same underlying incident.

        stopped_vehicle and slow_vehicle are one story - a vehicle not moving
        as it should. Without this, a truck that stops, is briefly re-read as
        crawling (box jitter around the stationary threshold), then stops again
        raises alerts under two different kinds for a single incident.
        """
        return "impeded_vehicle" if kind in {"stopped_vehicle", "slow_vehicle"} else kind

    def _is_duplicate(self, kind: str, bbox) -> bool:
        """True if a recent event about the same object and incident already fired."""
        cutoff = self._now - self.cfg.duplicate_window_s
        self._recent_alerts = [a for a in self._recent_alerts if a[0] >= cutoff]
        family = self._kind_family(kind)
        return any(self._kind_family(k) == family and self._iou(bbox, b) >= self.cfg.duplicate_iou
                   for _t, b, k in self._recent_alerts)

    def _can_fire(self, st: TrackState) -> bool:
        if st.vlm_calls >= self.cfg.max_calls_per_track:
            return False
        if st.last_trigger_s is not None and (self._now - st.last_trigger_s) < self.cfg.cooldown_seconds:
            return False
        return True

    # -- main ingest --------------------------------------------------------
    def update(self, det) -> list[Event]:
        """Ingest one frame of tracked detections; return any triggered events."""
        self._now = det.timestamp_s
        self.frames_seen += 1
        live_ids: set[int] = set()

        for i, tid in enumerate(det.track_ids):
            tid = int(tid)
            live_ids.add(tid)
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            diag = float(np.hypot(x2 - x1, y2 - y1))
            name = det.class_names[i] if i < len(det.class_names) else "object"

            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(
                    track_id=tid,
                    class_name=name,
                    first_seen_s=det.timestamp_s,
                    last_seen_s=det.timestamp_s,
                )
                self.tracks[tid] = st

            st.last_seen_s = det.timestamp_s
            st.class_name = name
            st.last_xyxy = (x1, y1, x2, y2)
            st.box_diag = diag
            st.frame_diag = getattr(det, "frame_diag", 0.0) or st.frame_diag
            st.history.append((det.timestamp_s, cx, cy, diag))
            st.zone_kind, zone = self.zones.lookup(cx, cy)
            st.zone_name = zone.name if zone else None
            self._update_kinematics(st)

            # Stationary bookkeeping. The latch is anchored, not speed-gated:
            # once a track stops we remember WHERE, and only release the latch
            # when it has actually travelled away from that spot. Releasing on
            # instantaneous speed let detector jitter reset a real 19s dwell.
            # "Stationary" requires stillness in BOTH the image plane and depth.
            depth_moving = st.scale_rate_med >= self.cfg.scale_rate_moving
            if st.stationary_since_s is None:
                if st.norm_speed < self.cfg.stationary_speed and not depth_moving:
                    st.stationary_since_s = det.timestamp_s
                    st.stop_anchor = (cx, cy)
            else:
                anchor = st.stop_anchor or (cx, cy)
                drift = float(np.hypot(cx - anchor[0], cy - anchor[1])) / max(diag, 1e-6)
                if drift > self.cfg.move_release_bodylengths or depth_moving:
                    st.stationary_since_s = None
                    st.stop_anchor = None

        # Needs every track's speed for this frame already updated, so it runs
        # here rather than inside the ingest loop above.
        self._update_slow_latch(live_ids, det)

        events: list[Event] = []
        for tid in live_ids:
            st = self.tracks[tid]
            if st.age_s < self.cfg.min_track_age_s or not self._can_fire(st):
                continue
            ev = self._evaluate(st, det)
            if ev is not None:
                if self._is_duplicate(ev.kind, ev.bbox):
                    # Same object, different track id - charge the cooldown so it
                    # does not retry every frame, but do not raise a second alert.
                    st.last_trigger_s = det.timestamp_s
                    continue
                st.vlm_calls += 1
                st.last_trigger_s = det.timestamp_s
                self._recent_alerts.append((det.timestamp_s, ev.bbox, ev.kind))
                self.events_fired += 1
                events.append(ev)

        crowd = self._check_crowd(det)
        if crowd is not None:
            self.events_fired += 1
            events.append(crowd)

        # Drop tracks unseen for a while so memory stays flat on long videos.
        stale = [tid for tid, st in self.tracks.items() if det.timestamp_s - st.last_seen_s > 30.0]
        for tid in stale:
            del self.tracks[tid]

        return events

    def _check_crowd(self, det) -> Event | None:
        """Scene-level rule: an unusual number of people gathered.

        Not per-track, so it carries its own cooldown rather than using the
        per-track budget. Density is judged in body-lengths, so it does not
        depend on altitude.
        """
        cfg = self.cfg
        people = [st for st in self.tracks.values()
                  if st.is_person and st.last_seen_s >= det.timestamp_s - 0.5]
        if len(people) < cfg.crowd_count:
            return None
        if self._last_crowd_s is not None and (det.timestamp_s - self._last_crowd_s) < cfg.crowd_cooldown_s:
            return None
        self._last_crowd_s = det.timestamp_s

        xs = [(p.last_xyxy[0] + p.last_xyxy[2]) / 2 for p in people]
        ys = [(p.last_xyxy[1] + p.last_xyxy[3]) / 2 for p in people]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        stationary = sum(1 for p in people if p.norm_speed < cfg.stationary_speed)
        ctx = (
            f"{len(people)} people are currently visible in {phrase_zone(people[0].zone_kind)}, "
            f"of whom {stationary} are stationary. "
            f"This is above the configured crowd threshold of {cfg.crowd_count}."
        )
        return Event(
            frame_idx=det.frame_idx,
            timestamp_s=det.timestamp_s,
            kind="crowd_density",
            track_id=None,
            class_name="person",
            zone_kind=people[0].zone_kind,
            bbox=(x1, y1, x2, y2),
            context=ctx,
            features={"person_count": len(people), "stationary_people": stationary,
                      "threshold": cfg.crowd_count},
            # A gathering is worth a look, not an automatic incident - the VLM
            # decides whether it is a queue, a market, or something wrong.
            rule_anomalous=False,
            rule_severity=0.3,
        )

    def _evaluate(self, st: TrackState, det) -> Event | None:
        """Apply the context rules. First match wins."""
        cfg = self.cfg
        lane_like = st.zone_kind in {"driving_lane", "unknown"}

        # Too small to judge motion on: a receding vehicle far up the road barely
        # moves in the image, so it looks stopped whatever its real speed.
        too_small = (st.frame_diag > 1.0
                     and st.box_diag / st.frame_diag < cfg.min_box_diag_frac)

        # Rule 1: vehicle stopped where vehicles should be moving.
        if st.is_vehicle and lane_like and not too_small and st.stationary_s >= cfg.stop_seconds:
            stopped, total = self._stopped_ratio(exclude_id=st.track_id)
            jam = total >= 3 and stopped / max(total, 1) >= 0.6
            flow_kmh, flow_ok = self._traffic_speed_kmh(exclude_id=st.track_id)
            # Only assert a speed the geometry actually supports.
            flow_txt = f" Surrounding traffic is moving at about {flow_kmh:.0f} km/h." if (
                flow_kmh and flow_ok) else ""
            ctx = (
                f"A {st.class_name} (track {st.track_id}) has been stationary for "
                f"{st.stationary_s:.0f}s in {phrase_zone(st.zone_kind)}. "
                f"It has been visible for {st.age_s:.0f}s. "
                f"Of the {total} other vehicles currently in view, {stopped} are also stopped"
                + (" - traffic appears congested overall." if jam else " - surrounding traffic is still flowing.")
                + flow_txt
            )
            return Event(
                frame_idx=det.frame_idx,
                timestamp_s=det.timestamp_s,
                kind="stopped_vehicle",
                track_id=st.track_id,
                class_name=st.class_name,
                zone_kind=st.zone_kind,
                bbox=st.last_xyxy,
                context=ctx,
                features={
                    "stationary_s": round(st.stationary_s, 1),
                    "age_s": round(st.age_s, 1),
                    "norm_speed": round(st.norm_speed, 4),
                    "neighbours_stopped": stopped,
                    "neighbours_total": total,
                    "congestion_suspected": jam,
                    "traffic_flow_kmh": round(flow_kmh, 1) if flow_kmh else None,
                    "traffic_flow_kmh_reliable": flow_ok,
                    "view_obliquity": round(self.view_obliquity(), 2) if self.view_obliquity() else None,
                    "metres_per_pixel": round(st.m_per_px, 5) if st.m_per_px else None,
                    "vehicle_length_ruler_m": EXPECTED_LENGTH_M.get(st.class_name),
                },
                # Congestion is a benign explanation; a lone stop in a live lane
                # is not. Severity grows with dwell and caps at 0.85.
                rule_anomalous=not jam,
                rule_severity=0.2 if jam else round(min(0.85, 0.55 + st.stationary_s / 120.0), 2),
            )

        # Rule 1b: vehicle crawling while the traffic around it flows normally.
        # Ordered after Rule 1 so a genuine stop always wins; this catches the
        # band between "stopped" and "normal", which the teacher flagged and the
        # dwell rule alone could not see.
        too_small_for_slow = (st.frame_diag > 1.0
                              and st.box_diag / st.frame_diag < cfg.slow_min_box_diag_frac)
        if (cfg.enable_slow_vehicle
                and st.is_vehicle and lane_like and not too_small_for_slow
                and st.slow_observations >= cfg.slow_min_observations
                and st.slow_fraction >= cfg.slow_fraction):
            ambient, n_moving = self._ambient_speed(exclude_id=st.track_id)
            if ambient:
                ratio = st.last_slow_ratio
                flow_kmh, flow_ok = self._traffic_speed_kmh(exclude_id=st.track_id)
                own_kmh = st.speed_kmh
                obliquity = self.view_obliquity()
                speed_ok = obliquity is not None and obliquity <= cfg.max_obliquity_for_speed
                if flow_kmh and flow_ok and own_kmh and speed_ok:
                    speed_txt = (f" It is doing about {own_kmh:.0f} km/h while surrounding "
                                 f"traffic moves at about {flow_kmh:.0f} km/h.")
                else:
                    speed_txt = (f" It is moving at {ratio * 100:.0f}% of the speed of "
                                 f"surrounding traffic.")
                ctx = (
                    f"A {st.class_name} (track {st.track_id}) has been moving abnormally "
                    f"slowly in {phrase_zone(st.zone_kind)} for "
                    f"{int(st.slow_fraction * 100)}% of the {st.slow_observations} "
                    f"observations of it, while {n_moving} other vehicles nearby are "
                    f"moving normally.{speed_txt}"
                )
                return Event(
                    frame_idx=det.frame_idx,
                    timestamp_s=det.timestamp_s,
                    kind="slow_vehicle",
                    track_id=st.track_id,
                    class_name=st.class_name,
                    zone_kind=st.zone_kind,
                    bbox=st.last_xyxy,
                    context=ctx,
                    features={
                        "slow_fraction": round(st.slow_fraction, 3),
                        "slow_observations": st.slow_observations,
                        "norm_speed": round(st.norm_speed, 4),
                        "ambient_speed": round(ambient, 4),
                        "speed_ratio": round(ratio, 3),
                        "moving_neighbours": n_moving,
                        "own_kmh": round(own_kmh, 1) if own_kmh else None,
                        "traffic_flow_kmh": round(flow_kmh, 1) if flow_kmh else None,
                        "kmh_reliable": bool(flow_ok and speed_ok),
                    },
                    # Less urgent than a dead stop, but a vehicle crawling in live
                    # traffic is a genuine hazard. Severity scales with how
                    # consistently slow it has been.
                    rule_anomalous=True,
                    # Capped below the stopped_vehicle range on purpose: a
                    # crawling vehicle is a lesser incident than a dead stop,
                    # and severity ordering should reflect that.
                    rule_severity=round(min(0.55, 0.3 + 0.25 * st.slow_fraction), 2),
                )

        # Rule 2: pedestrian in the roadway.
        if st.is_person and st.zone_kind == "driving_lane" and st.age_s >= cfg.person_in_lane_seconds:
            ctx = (
                f"A person (track {st.track_id}) has been in a live driving lane for "
                f"{st.age_s:.0f}s, moving at {st.norm_speed:.2f} body-lengths/sec."
            )
            return Event(
                frame_idx=det.frame_idx,
                timestamp_s=det.timestamp_s,
                kind="person_in_roadway",
                track_id=st.track_id,
                class_name=st.class_name,
                zone_kind=st.zone_kind,
                bbox=st.last_xyxy,
                context=ctx,
                features={"age_s": round(st.age_s, 1), "norm_speed": round(st.norm_speed, 4)},
                rule_anomalous=True,
                rule_severity=0.85,  # a pedestrian on a carriageway is unambiguous
            )

        # Rule 3: a person stationary far longer than normal movement implies.
        # Class-agnostic dwell is what lets the same architecture carry over from
        # a highway to a plaza with no code change - only the class differs.
        if st.is_person and st.stationary_s >= cfg.loiter_seconds and st.zone_kind != "sidewalk":
            severe = st.zone_kind in {"driving_lane", "restricted"}
            ctx = (
                f"A person (track {st.track_id}) has remained stationary for "
                f"{st.stationary_s:.0f}s in {phrase_zone(st.zone_kind)}, "
                f"having been visible for {st.age_s:.0f}s."
            )
            return Event(
                frame_idx=det.frame_idx,
                timestamp_s=det.timestamp_s,
                kind="loitering",
                track_id=st.track_id,
                class_name=st.class_name,
                zone_kind=st.zone_kind,
                bbox=st.last_xyxy,
                context=ctx,
                features={"stationary_s": round(st.stationary_s, 1), "age_s": round(st.age_s, 1)},
                rule_anomalous=severe,
                rule_severity=0.7 if severe else 0.35,
            )

        # Rule 4: vehicle travelling against the calibrated flow of its lane.
        if st.is_vehicle and st.heading_deg is not None and st.norm_speed >= cfg.wrong_way_min_speed:
            _, zone = self.zones.lookup(*st.history[-1][1:3])
            if zone is not None and zone.flow_deg is not None:
                delta = abs((st.heading_deg - zone.flow_deg + 180.0) % 360.0 - 180.0)
                if delta >= cfg.wrong_way_tolerance_deg:
                    ctx = (
                        f"A {st.class_name} (track {st.track_id}) is travelling at "
                        f"{st.heading_deg:.0f}deg in lane '{zone.name}' whose expected flow is "
                        f"{zone.flow_deg:.0f}deg - a {delta:.0f}deg deviation."
                    )
                    return Event(
                        frame_idx=det.frame_idx,
                        timestamp_s=det.timestamp_s,
                        kind="wrong_way_vehicle",
                        track_id=st.track_id,
                        class_name=st.class_name,
                        zone_kind=st.zone_kind,
                        bbox=st.last_xyxy,
                        context=ctx,
                        features={"heading_deg": round(st.heading_deg, 1), "deviation_deg": round(delta, 1)},
                        rule_anomalous=True,
                        rule_severity=0.9,
                    )
        return None

    def sample_normal(self, det, rng=None) -> Event | None:
        """Emit a `normal_sample` event for a track that is behaving normally.

        Distillation needs negatives. Triggered events on free-flowing footage
        are almost all genuine anomalies, which yields a single-class label set
        that teaches no decision boundary. Sampling ordinary moving traffic
        gives the teacher something to label benign, in the same context format
        the student sees at inference.
        """
        import random

        rng = rng or random
        candidates = [
            st
            for st in self.tracks.values()
            if st.last_seen_s >= det.timestamp_s - 0.01
            and st.age_s >= self.cfg.min_track_age_s
            and st.norm_speed >= self.cfg.stationary_speed
            and st.vlm_calls == 0
        ]
        if not candidates:
            return None
        st = rng.choice(candidates)
        st.vlm_calls += 1  # sample each track at most once
        stopped, total = self._stopped_ratio(exclude_id=st.track_id)
        obliquity = self.view_obliquity()
        speed_ok = obliquity is not None and obliquity <= self.cfg.max_obliquity_for_speed
        speed_txt = (f"about {st.speed_kmh:.0f} km/h" if (st.speed_kmh and speed_ok)
                     else f"{st.norm_speed:.2f} body-lengths/sec")
        ctx = (
            f"A {st.class_name} (track {st.track_id}) is moving at {speed_txt} in "
            f"{phrase_zone(st.zone_kind)}. "
            f"It has been visible for {st.age_s:.0f}s and has not been stationary. "
            f"Of the {total} other vehicles currently in view, {stopped} are stopped."
        )
        return Event(
            frame_idx=det.frame_idx,
            timestamp_s=det.timestamp_s,
            kind="normal_sample",
            track_id=st.track_id,
            class_name=st.class_name,
            zone_kind=st.zone_kind,
            bbox=st.last_xyxy,
            context=ctx,
            features={
                "norm_speed": round(st.norm_speed, 4),
                "age_s": round(st.age_s, 1),
                "neighbours_stopped": stopped,
                "neighbours_total": total,
                "sampled_as_negative": True,
            },
            rule_anomalous=False,
            rule_severity=0.1,
        )

    def stats(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "events_fired": self.events_fired,
            "trigger_rate_pct": round(100.0 * self.events_fired / max(self.frames_seen, 1), 2),
            "live_tracks": len(self.tracks),
        }


# --- Self-test on synthetic tracks ----------------------------------------
@dataclass
class _FakeDet:
    """Mimics detect_track.FrameDetections without needing a model or video."""

    frame_idx: int
    timestamp_s: float
    track_ids: np.ndarray
    xyxy: np.ndarray
    class_names: list[str]


def _selftest() -> int:
    fps = 10.0
    cfg = TriggerConfig(stop_seconds=5.0, cooldown_seconds=1e9, min_track_age_s=0.5)
    zones = ZoneMap(
        [Zone("lane1", "driving_lane", np.array([[0, 0], [1000, 0], [1000, 600], [0, 600]], np.int32), flow_deg=0.0)]
    )
    tracker = ContextStateTracker(zones=zones, config=cfg)

    fired: list[Event] = []
    for f in range(120):  # 12 seconds
        t = f / fps
        # Track 1: parked at x=100 the whole time -> must fire.
        box_stopped = [100, 300, 160, 340]
        # Track 2: cruising left-to-right -> must never fire.
        x = 200 + 12 * f
        box_moving = [x, 400, x + 60, 440]
        det = _FakeDet(
            frame_idx=f,
            timestamp_s=t,
            track_ids=np.array([1, 2]),
            xyxy=np.array([box_stopped, box_moving], dtype=float),
            class_names=["car", "car"],
        )
        fired.extend(tracker.update(det))

    stopped_events = [e for e in fired if e.track_id == 1]
    moving_events = [e for e in fired if e.track_id == 2]

    failures: list[str] = []
    if not stopped_events:
        failures.append("stationary vehicle (track 1) never triggered")
    if moving_events:
        failures.append(f"moving vehicle (track 2) wrongly triggered {len(moving_events)}x")
    if stopped_events:
        first = stopped_events[0]
        if first.kind != "stopped_vehicle":
            failures.append(f"expected kind 'stopped_vehicle', got {first.kind!r}")
        # Fires once the dwell threshold is crossed, not before.
        if first.timestamp_s < cfg.stop_seconds - 0.5:
            failures.append(f"fired too early at {first.timestamp_s:.1f}s (threshold {cfg.stop_seconds}s)")
        if "still flowing" not in first.context:
            failures.append("context should note surrounding traffic is flowing")

    # Wrong-way: a car driving right-to-left against flow_deg=0.
    tracker2 = ContextStateTracker(zones=zones, config=TriggerConfig(stop_seconds=1e9, min_track_age_s=0.5))
    wrong_way: list[Event] = []
    for f in range(60):
        x = 900 - 14 * f
        det = _FakeDet(
            frame_idx=f,
            timestamp_s=f / fps,
            track_ids=np.array([9]),
            xyxy=np.array([[x, 200, x + 60, 240]], dtype=float),
            class_names=["car"],
        )
        wrong_way.extend(tracker2.update(det))
    if not any(e.kind == "wrong_way_vehicle" for e in wrong_way):
        failures.append("wrong-way vehicle never triggered")

    print("=== Stage 2 self-test ===")
    print(f"stopped-vehicle events : {len(stopped_events)}")
    print(f"moving-vehicle events  : {len(moving_events)} (expect 0)")
    print(f"wrong-way events       : {sum(1 for e in wrong_way if e.kind == 'wrong_way_vehicle')}")
    if stopped_events:
        print(f"first trigger at       : {stopped_events[0].timestamp_s:.1f}s")
        print(f"context sent to VLM    : {stopped_events[0].context}")
    print(f"trigger rate           : {tracker.stats()['trigger_rate_pct']}% of frames")

    if failures:
        print("\nFAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nPASS - all Stage 2 assertions held.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: per-track context state and event triggers.")
    p.add_argument("--selftest", action="store_true", help="Run assertions against synthetic tracks.")
    p.add_argument("--data_dir", default=None, help="Unused here; kept for interface parity.")
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    p.print_help()


if __name__ == "__main__":
    main()
