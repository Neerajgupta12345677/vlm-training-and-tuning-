"""Maps our internal event vocabulary onto the organisers' exact 12-class schema.

Pure logic, no I/O, so it can be unit-tested without the real dataset. The
organisers' doc says class_name must "match exactly" - a silent typo here
would cost accuracy on every single video, invisibly.

Official label set (from "AHC Visual Intelligence Hackathon - Training and
Public Test Data", verbatim):
    normal, traffic_accident, traffic_congestion,
    stalled_or_broken_down_vehicle, vehicle_blocking_traffic,
    wrong_way_driving, road_spill_or_debris, waterlogging_or_flood,
    fire, smoke, fighting_or_violence, loitering_or_suspicious_presence
"""

from __future__ import annotations

OFFICIAL_LABELS = frozenset({
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
})

# Our Stage-2 rule kind -> official label. Only rules that fire with
# rule_anomalous=True are mapped here; rules that are benign by default
# (crowd_density) are handled separately in map_event_to_label below, because
# their label depends on WHY they became anomalous, not on the rule name.
_KIND_TO_LABEL = {
    # A crashed car is also a stopped car, so the stop alone cannot choose
    # between these two. collision_signature is the rule that measures HOW the
    # stop happened (several vehicles, at once, in flowing traffic).
    "collision_signature": "traffic_accident",
    "stopped_vehicle": "stalled_or_broken_down_vehicle",
    "slow_vehicle": "vehicle_blocking_traffic",
    "wrong_way_vehicle": "wrong_way_driving",
    "traffic_congestion": "traffic_congestion",
    "loitering": "loitering_or_suspicious_presence",
}

# vlm_reason._REAL_HAZARDS keyword -> official label. This is the path for
# every event the tracker cannot see (fire/smoke/flood/debris/collision/fight) -
# the label comes from what the VLM actually named, not from a Stage-2 rule.
_HAZARD_TO_LABEL = {
    "fire": "fire",
    "smoke": "smoke",
    "flood": "waterlogging_or_flood",
    "water": "waterlogging_or_flood",
    "waterlog": "waterlogging_or_flood",
    "submerg": "waterlogging_or_flood",
    "debris": "road_spill_or_debris",
    "spill": "road_spill_or_debris",
    "collision": "traffic_accident",
    "crash": "traffic_accident",
    "explosion": "traffic_accident",
    "fight": "fighting_or_violence",
    "violence": "fighting_or_violence",
    "assault": "fighting_or_violence",
    # "crowd" deliberately has no entry: see map_event_to_label. A dense crowd
    # is not itself one of the twelve labels, and forcing it onto
    # loitering_or_suspicious_presence would manufacture false positives on
    # every legitimate gathering (a bus terminal, a market) the sweep sees.
}

# Two of our detections have NO clean official equivalent. Documented here
# rather than silently guessed, so the approximation is visible and can be
# revisited once the real submission format / rubric is confirmed.
#
#   person_in_roadway: closest is loitering_or_suspicious_presence (an
#     unexpected person presence), but it is a weaker match than every other
#     row in this table - a person in a live lane is arguably closer to a
#     safety hazard than "suspicious presence" implies. Kept mapped rather
#     than dropped, because an approximate label beats silence on a metric
#     that likely rewards recall.
#   crowd_density (unescalated): has no official label at all and is
#     EXCLUDED from submission rows entirely - see map_event_to_label.
#   MEASURED 2026-09-04 on the organisers' public test set: this mapping
#   produced 3 false positives and 0 true positives (T011, T025, T026 - none
#   of which are loitering clips). A person crossing a carriageway is not
#   "suspicious presence", and the approximation cost precision on a benchmark
#   that weights false alarms as heavily as misses. Disabled rather than
#   deleted so the reasoning stays visible if real footage argues for it back.
_APPROXIMATE: dict[str, str] = {
    # "person_in_roadway": "loitering_or_suspicious_presence",
}


def hazard_to_label(hazard_type: str) -> str | None:
    """Map a vlm_reason hazard_type string to an official label, or None."""
    t = (hazard_type or "").strip().lower()
    for keyword, label in _HAZARD_TO_LABEL.items():
        if keyword in t:
            return label
    return None


def map_event_to_label(kind: str, hazard_type: str | None = None) -> str | None:
    """The official label for one fired Event, or None if it should not be
    submitted as its own row (currently: an unescalated crowd_density).

    `hazard_type` is the VLM's hazard_type when the event carries an
    `observation` (hybrid decision mode) - it takes priority over the rule
    kind, because a scene_sweep event's kind is not itself a label, and a
    VLM-escalated crowd_density's real label is whatever hazard was seen
    (e.g. "fight"), not "crowd".
    """
    if hazard_type:
        mapped = hazard_to_label(hazard_type)
        if mapped:
            return mapped
        # hazard_type was set but didn't match a real hazard (e.g. "crowd",
        # "none", or garbage) - fall through to the rule-based mapping below,
        # since _is_real_hazard() in vlm_reason already filtered out garbage
        # before this event could have rule_anomalous forced True by a hazard.

    if kind in _KIND_TO_LABEL:
        return _KIND_TO_LABEL[kind]
    if kind in _APPROXIMATE:
        return _APPROXIMATE[kind]
    if kind == "crowd_density":
        # Only reachable here if not VLM-escalated (no matching hazard above).
        # A crowd on its own is not one of the twelve labels - submitting it
        # as anything would be inventing a false positive class, not
        # approximating a real one.
        return None
    if kind in {"scene_sweep", "normal_sample"}:
        # scene_sweep alone carries no verdict of its own (see
        # context_state._check_scene_sweep - rule_anomalous is always False);
        # its escalations are handled by the hazard_type branch above.
        # normal_sample only exists for harvesting distillation negatives and
        # is never a real detection.
        return None
    return None  # unknown kind: fail closed, never guess a label


def validate_label(label: str) -> str:
    """Raise if `label` is not one of the exact official strings.

    Call this at the submission-writer boundary - the one place a typo would
    otherwise silently cost accuracy on every row using it.
    """
    if label not in OFFICIAL_LABELS:
        raise ValueError(
            f"{label!r} is not an official label. Valid: {sorted(OFFICIAL_LABELS)}"
        )
    return label
