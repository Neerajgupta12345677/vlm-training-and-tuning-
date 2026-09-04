"""STAGE 3 - event-triggered only: ask a small VLM for a structured verdict.

Runtime is Ollama (native Windows CUDA, no build step). The model stays in
Ollama's process, which keeps its weights out of our Python RAM budget.

Two techniques that matter on a 4GB card:
  * The target object gets a drawn highlight box before the frame is sent, so
    the model knows which object we mean instead of inferring it from text.
  * Frames are downscaled to --max-side, which cuts both latency and tokens.

A `mock` backend implements the same interface with no model at all, so the
pipeline is runnable (and demoable) with zero weights and no network.

    python src\\vlm_reason.py --selftest --backend mock
    python src\\vlm_reason.py --selftest --backend ollama --model moondream
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common import (
    OUTPUTS_DIR,
    AnomalyVerdict,
    SceneObservation,
    observation_json_schema,
    verdict_json_schema,
)

OLLAMA_URL = "http://localhost:11434"

# Division of labour matters here. A single still frame cannot reveal motion - a
# moving car and a stopped car are pixel-identical in one image. Measured: when
# asked to judge motion, qwen2.5vl:3b called all 6 test frames anomalous (3/6,
# chance), reporting "a vehicle is stopped" for vehicles that were moving.
# The tracker already knows motion with certainty. So the prompt orders the VLM
# to TRUST those facts and rules on what it genuinely can see: the surroundings.
SYSTEM_PROMPT = (
    "You are a drone surveillance analyst reviewing one frame of aerial video.\n"
    "The object under review is outlined by a bright magenta box.\n\n"
    "A motion tracker has ALREADY measured whether that object is moving or stationary, "
    "and for how long. Those measured facts are given to you and are RELIABLE. Do not "
    "second-guess them - you cannot judge motion from a single still frame.\n\n"
    "Your job is to rule on what the tracker cannot see: the surroundings.\n\n"
    "Apply these rules in order and stop at the first one that matches:\n"
    "1. You can see fire, smoke, a collision, spilled debris, a person standing on the "
    "carriageway, or a crowd gathering -> anomalous = true, severity 0.9\n"
    "2. Tracker says STATIONARY, it is in a live traffic lane, and the tracker reports "
    "that the other vehicles are still flowing -> a broken-down or abandoned vehicle "
    "blocking live traffic -> anomalous = true, severity 0.7\n"
    "3. Tracker says STATIONARY and the tracker reports most other vehicles are stopped "
    "too -> ordinary congestion -> anomalous = false, severity 0.2\n"
    "4. Tracker says STATIONARY in a parking area, lay-by or hard shoulder -> a normal "
    "place to stop -> anomalous = false, severity 0.1\n"
    "5. Tracker says the object is MOVING -> normal traffic -> anomalous = false, "
    "severity 0.1\n\n"
    "Rule 2 is the single most important case in this system: a vehicle stopped in a live "
    "lane while other traffic flows around it is exactly the incident operators need to "
    "know about. Report it as anomalous.\n\n"
    "Answer with a single JSON object and nothing else. The example below describes an "
    "unrelated scene and is here ONLY to show the shape - never reuse its wording:\n"
    '{"anomalous": true, "severity": 0.5, "reason": "Livestock has strayed onto a rural '
    'level crossing."}\n'
    "severity is 0.0-1.0: below 0.3 benign, 0.3-0.5 worth logging, 0.6-0.8 dispatch an "
    "operator, above 0.9 emergency. The reason must be one sentence about what YOU see in "
    "THIS image - never empty, never copied from the example."
)

# No literal placeholder text here: given a template containing "short sentence",
# small models copy that string straight into the reason field.
SIMPLE_RETRY_PROMPT = (
    "Look at the object outlined in magenta. Reply with only a JSON object having three "
    "keys: anomalous (true or false), severity (a number from 0.0 to 1.0), and reason. "
    "For reason, write one sentence describing what you actually see happening in this "
    "image. Do not reuse the wording of this instruction."
)


OBSERVE_SYSTEM = (
    "You are a drone imagery observer. Report only what is visibly present in the frame.\n"
    "The object of interest is outlined by a bright magenta box.\n"
    "Do not judge whether anything is moving - you cannot tell that from one still frame, "
    "and you are not being asked to.\n"
    "Report three things:\n"
    "  hazard_visible: true only if you can actually SEE one of the hazards listed "
    "below. Otherwise false.\n"
    "  hazard_type: must be exactly one word from this list: fire, smoke, collision, "
    "debris, flood, fight, crowd, none.\n"
    "    fire/smoke  - visible flames or a smoke plume\n"
    "    collision   - vehicles crashed into each other or overturned\n"
    "    debris      - spilled load, rubble or objects lying in the roadway\n"
    "    flood       - standing water or a submerged road surface\n"
    "    fight       - people physically fighting\n"
    "    crowd       - a dense crowd gathered where one would not be expected\n"
    "  Use 'none' for anything else, including ordinary people, ordinary traffic, "
    "parked vehicles, wet-but-passable road, or shadows. Ordinary is the common case.\n"
    "  surroundings: one short sentence saying where the boxed object sits (live traffic "
    "lane, hard shoulder, lay-by, parking area, junction) and what is immediately around it.\n"
    "Answer with a single JSON object and nothing else."
)

OBSERVE_PROMPT = (
    "Look at the object in the magenta box. Report hazard_visible, hazard_type and "
    "surroundings for this frame as JSON."
)


def build_prompt_for_training(context: str) -> str:
    """The user-turn text for a given context.

    Single source of truth: `judge()` sends this at inference time and
    build_kaggle_dataset.py bakes the identical string into the training rows,
    so the student is never trained on wording it won't see in production.
    """
    return (
        f"Measured tracker facts about the object in the magenta box (these are "
        f"reliable - trust them):\n{context}\n\n"
        "Taking those facts as given, and judging from what you can see of the "
        "surroundings in this frame, is this a genuine anomaly an operator must act on?"
    )


@dataclass
class VLMResult:
    verdict: AnomalyVerdict
    latency_ms: float
    backend: str
    model: str
    attempts: int
    raw: str = ""
    parse_failed_once: bool = False


def highlight_target(frame: np.ndarray, bbox: tuple[float, float, float, float] | None) -> np.ndarray:
    """Draw a magenta box around the flagged object (visual prompting)."""
    out = frame.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    thickness = max(2, int(round(min(h, w) / 240)))
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), thickness)
    return out


def downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg_b64(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_verdict(text: str) -> AnomalyVerdict:
    """Parse a verdict, tolerating prose or code fences around the JSON."""
    text = text.strip()
    try:
        return AnomalyVerdict.model_validate_json(text)
    except Exception:  # noqa: BLE001 - fall through to salvage attempts
        pass
    # Salvage the first {...} block; small models like to add commentary.
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        blob = match.group(0)
        try:
            return AnomalyVerdict.model_validate_json(blob)
        except Exception:  # noqa: BLE001
            data = json.loads(blob)  # may raise; caller handles
            return AnomalyVerdict(
                anomalous=bool(data.get("anomalous", False)),
                severity=float(np.clip(float(data.get("severity", 0.0)), 0.0, 1.0)),
                reason=str(data.get("reason", ""))[:300],
            )
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


class VLMReasoner:
    def __init__(
        self,
        backend: str = "ollama",
        model: str = "moondream",
        max_side: int = 768,
        timeout_s: float = 120.0,
        url: str = OLLAMA_URL,
        constrain_first: bool = False,
    ) -> None:
        self.backend = backend
        self.model = model
        self.max_side = max_side
        self.timeout_s = timeout_s
        self.url = url
        # Off by default: schema-constrained decoding measurably degrades small
        # models here. Turn on only if a model proves it needs the guardrail.
        self.constrain_first = constrain_first

    # -- backends -----------------------------------------------------------
    def _call_ollama(self, image_b64: str, prompt: str, constrain: bool,
                     system: str | None = None, schema: dict | None = None) -> str:
        import requests  # imported lazily so `mock` works without the dep

        payload: dict = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system or SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
            "options": {"temperature": 0.1, "num_predict": 160},
        }
        if schema is not None:
            payload["format"] = schema
        elif constrain:
            # Ollama constrains decoding to this schema, which is what makes a
            # 0.5-2B model reliably emit parseable JSON.
            payload["format"] = verdict_json_schema()

        resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_mock(self, context: str) -> str:
        """Deterministic stand-in with the same contract - no model, no network."""
        ctx = context.lower()
        stationary = re.search(r"stationary for (\d+)s", ctx)
        congested = "congested" in ctx
        if "person" in ctx and "driving lane" in ctx:
            return json.dumps(
                {"anomalous": True, "severity": 0.85, "reason": "Pedestrian in a live traffic lane."}
            )
        if "wrong" in ctx or "deviation" in ctx:
            return json.dumps(
                {"anomalous": True, "severity": 0.9, "reason": "Vehicle travelling against lane flow."}
            )
        if stationary:
            secs = int(stationary.group(1))
            if congested:
                return json.dumps(
                    {"anomalous": False, "severity": 0.2, "reason": "Vehicle stopped in general congestion."}
                )
            sev = float(np.clip(0.4 + secs / 120.0, 0.0, 1.0))
            return json.dumps(
                {
                    "anomalous": True,
                    "severity": round(sev, 2),
                    "reason": f"Vehicle stopped {secs}s in a live lane while traffic flows.",
                }
            )
        return json.dumps({"anomalous": False, "severity": 0.1, "reason": "No anomaly evident."})

    def prepare_image(self, frame: np.ndarray, bbox=None) -> str:
        """Highlight + downscale + JPEG-encode. Cheap (~5ms), runs synchronously.

        Doing this before handing work to a background thread means the queue
        holds ~40KB JPEGs instead of 25MB raw 4K frames.
        """
        return encode_jpeg_b64(downscale(highlight_target(frame, bbox), self.max_side))

    def observe(self, frame: np.ndarray, bbox=None) -> tuple[SceneObservation, float, str]:
        """Ask only what the model can see. Returns (observation, ms, raw)."""
        if self.backend == "mock":
            return self.observe_b64("")
        return self.observe_b64(self.prepare_image(frame, bbox))

    def observe_b64(self, image_b64: str) -> tuple[SceneObservation, float, str]:
        """Observation on an already-encoded image, so it is thread-submittable.

        This is the reliable half of Stage 3. Schema-constrained decoding is
        safe here because the task is descriptive, not a conditional judgement.
        """
        t0 = time.perf_counter()
        if self.backend == "mock":
            obs = SceneObservation(
                hazard_visible=False, hazard_type="none",
                surroundings="A multi-lane carriageway with traffic in adjacent lanes.",
            )
            return obs, (time.perf_counter() - t0) * 1000.0, "(mock)"

        raw = self._call_ollama(image_b64, OBSERVE_PROMPT, constrain=False,
                                system=OBSERVE_SYSTEM, schema=observation_json_schema())
        try:
            obs = SceneObservation.model_validate_json(raw.strip())
        except Exception:  # noqa: BLE001
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            try:
                obs = SceneObservation.model_validate_json(match.group(0)) if match else None
            except Exception:  # noqa: BLE001
                obs = None
            if obs is None:
                obs = SceneObservation(
                    hazard_visible=False, hazard_type="none",
                    surroundings="(observation unavailable)",
                )
        return obs, (time.perf_counter() - t0) * 1000.0, raw

    # -- public API ---------------------------------------------------------
    def judge(
        self,
        frame: np.ndarray,
        context: str,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> VLMResult:
        if self.backend == "mock":
            t0 = time.perf_counter()
            raw = self._call_mock(context)
            return VLMResult(
                verdict=parse_verdict(raw),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                backend="mock",
                model="mock",
                attempts=1,
                raw=raw,
            )
        return self.judge_b64(self.prepare_image(frame, bbox), context)

    def judge_b64(self, image: str, context: str) -> VLMResult:
        """Full judgement on an already-encoded image, so it is thread-submittable."""
        prompt = build_prompt_for_training(context)
        t0 = time.perf_counter()

        if self.backend == "mock":
            raw = self._call_mock(context)
            return VLMResult(
                verdict=parse_verdict(raw),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                backend="mock",
                model="mock",
                attempts=1,
                raw=raw,
            )

        # Attempt 1 is deliberately UNCONSTRAINED. Measured on moondream: with
        # Ollama's JSON-schema constraint the model collapses to a minimally
        # valid answer ({"anomalous": false, "severity": 0, "reason": ""}) and
        # gets the verdict wrong; unconstrained on the same frame it reasons
        # correctly. The schema is therefore a safety net (attempt 2), not the
        # default path. parse_verdict() already salvages JSON out of prose.
        raw = self._call_ollama(image, prompt, constrain=self.constrain_first)
        try:
            verdict = parse_verdict(raw)
            if verdict.reason.strip():  # an empty reason is the collapse signature
                return VLMResult(
                    verdict=verdict,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    backend=self.backend,
                    model=self.model,
                    attempts=1,
                    raw=raw,
                )
        except Exception:  # noqa: BLE001 - fall through to the constrained retry
            pass

        raw2 = self._call_ollama(image, SIMPLE_RETRY_PROMPT, constrain=True)
        try:
            verdict = parse_verdict(raw2)
        except Exception:  # noqa: BLE001 - never let Stage 3 crash the pipeline
            verdict = AnomalyVerdict(
                anomalous=False,
                severity=0.0,
                reason="VLM output unparseable after retry; treated as non-anomalous.",
            )
        return VLMResult(
            verdict=verdict,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.backend,
            model=self.model,
            attempts=2,
            raw=raw2,
            parse_failed_once=True,
        )


# The only hazard categories allowed to trigger an escalation. Measured bug:
# moondream reported hazard_type="person" for an ordinary pedestrian scene (a
# crowd is not, by itself, a hazard), and the old check - "anything that isn't
# literally 'none'" - accepted it and fired a false 0.9-severity alert on a
# normal crowd. A weak model's hallucinated noise must not be able to trigger
# the escalation path; only these genuinely unambiguous visual hazards can.
_REAL_HAZARDS = (
    "fire", "smoke", "collision", "crash", "explosion", "debris", "crowd",
    # Static conditions from the organisers' event list. These have no motion
    # signature at all, so the tracker can never find them - the scene sweep
    # plus these keywords is the only path by which they can be reported.
    "flood", "water", "spill", "waterlog", "submerg", "drain",
    "fight", "violence", "assault",
)


def _is_real_hazard(hazard_type: str) -> bool:
    t = (hazard_type or "").strip().lower()
    return any(h in t for h in _REAL_HAZARDS)


def combine(event_kind: str, rule_anomalous: bool, rule_severity: float,
            obs: SceneObservation) -> AnomalyVerdict:
    """Fuse Stage 2's deterministic verdict with the VLM's observation.

    Division of labour, chosen from measurements rather than taste:
      * Stage 2 owns the boolean. Dwell time, zone and neighbour state are
        arithmetic - a small VLM cannot beat them and measurably made them worse.
      * The VLM can only ESCALATE, never silently overturn: a visible hazard
        promotes any event to anomalous at high severity. A model that says
        "nothing visible" cannot clear a stop that the tracker measured.
      * Escalation requires hazard_type to match a fixed vocabulary
        (_REAL_HAZARDS), not merely be non-empty - see the bug note above.
    """
    if obs.hazard_visible and _is_real_hazard(obs.hazard_type):
        return AnomalyVerdict(
            anomalous=True,
            severity=max(0.9, rule_severity),
            reason=f"{obs.hazard_type.capitalize()} visible at the scene. {obs.surroundings}".strip(),
        )

    if rule_anomalous:
        if event_kind == "stopped_vehicle":
            lead = "Vehicle stopped in live traffic while surrounding vehicles keep moving."
        elif event_kind == "person_in_roadway":
            lead = "Person detected in a live traffic lane."
        elif event_kind == "wrong_way_vehicle":
            lead = "Vehicle travelling against the direction of its lane."
        else:
            lead = "Tracker rule flagged this object."
        return AnomalyVerdict(anomalous=True, severity=rule_severity,
                              reason=f"{lead} {obs.surroundings}".strip())

    reason = "Behaviour consistent with normal traffic."
    if event_kind == "stopped_vehicle":
        reason = "Vehicle stopped, but surrounding traffic is stopped too - congestion, not an incident."
    return AnomalyVerdict(anomalous=False, severity=rule_severity,
                          reason=f"{reason} {obs.surroundings}".strip())


def check_ollama(url: str = OLLAMA_URL) -> tuple[bool, list[str]]:
    """Return (reachable, installed model names)."""
    try:
        import requests

        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return False, []


def _load_real_event(events_path: Path, frames_dir: Path):
    """Return (frame, bbox, context) from a harvested event, or None."""
    if not events_path.exists():
        return None
    import json

    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "stopped_vehicle":
            continue
        frame = cv2.imread(str(frames_dir / row["frame_file"]))
        if frame is not None:
            return frame, tuple(row["bbox"]), row["context"], row["frame_file"]
    return None


def _selftest(args) -> int:
    """Judge a real harvested event frame if one exists, else a synthetic one.

    Testing on synthetic shapes is actively misleading: moondream described a
    grey/red rectangle scene as "urn of purple liquid", which says nothing about
    its performance on the actual drone footage it will be asked to judge.
    """
    real = _load_real_event(
        Path(args.events) if args.events else OUTPUTS_DIR / "harvest_events.jsonl",
        Path(args.frames_dir) if args.frames_dir else OUTPUTS_DIR / "events",
    )
    if real is not None:
        frame, bbox, context, src_name = real
        print(f"[input] real event frame {src_name} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        frame = np.full((480, 854, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (0, 300), (854, 400), (90, 90, 90), -1)  # a "road"
        cv2.rectangle(frame, (380, 320), (450, 370), (200, 40, 40), -1)  # a "car"
        bbox = (380, 320, 450, 370)
        context = (
            "A car (track 7) has been stationary for 34s in a live driving lane. "
            "It has been visible for 41s. Of the 6 other vehicles currently in view, "
            "0 are also stopped - surrounding traffic is still flowing."
        )
        print("[input] SYNTHETIC frame (no harvested events found) - results are not "
              "representative of real footage")
    print(f"[context] {context}")

    if args.backend == "ollama":
        ok, models = check_ollama(args.url)
        if not ok:
            print(f"[fail] Ollama not reachable at {args.url}. Is it running?")
            return 1
        print(f"[ok] Ollama reachable. Models: {models or '(none installed)'}")
        if not any(args.model in m for m in models):
            print(f"[fail] model {args.model!r} not installed. Run: ollama pull {args.model}")
            return 1

    reasoner = VLMReasoner(
        backend=args.backend,
        model=args.model,
        max_side=args.max_side,
        timeout_s=args.timeout,
        url=args.url,
        constrain_first=args.constrain_first,
    )
    res = reasoner.judge(frame, context, bbox)

    print("=== Stage 3 self-test ===")
    print(f"backend/model : {res.backend} / {res.model}")
    print(f"latency       : {res.latency_ms:.0f} ms")
    print(f"attempts      : {res.attempts}{'  (first parse failed)' if res.parse_failed_once else ''}")
    print(f"anomalous     : {res.verdict.anomalous}")
    print(f"severity      : {res.verdict.severity}")
    print(f"reason        : {res.verdict.reason}")
    if args.backend != "mock":
        print(f"raw           : {res.raw[:200]}")
    print("\nPASS - a schema-valid verdict was produced.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: VLM reasoning on a triggered event.")
    p.add_argument("--backend", default="ollama", choices=["ollama", "mock"])
    p.add_argument("--model", default="moondream", help="Ollama model tag, e.g. moondream / qwen2.5vl:3b")
    p.add_argument("--url", default=OLLAMA_URL)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--data_dir", default=None, help="Unused here; kept for interface parity.")
    p.add_argument("--events", default=None, help="Events jsonl to pull a real test frame from.")
    p.add_argument("--frames-dir", default=None, help="Where those event frames live.")
    p.add_argument("--constrain-first", action="store_true",
                   help="Use JSON-schema-constrained decoding on the first attempt "
                        "(measurably worse on small models; off by default).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--list", action="store_true", help="List Ollama models and exit.")
    args = p.parse_args()

    if args.list:
        ok, models = check_ollama(args.url)
        print(f"reachable: {ok}")
        for m in models:
            print(f"  {m}")
        return
    if args.selftest:
        raise SystemExit(_selftest(args))
    p.print_help()


if __name__ == "__main__":
    main()
