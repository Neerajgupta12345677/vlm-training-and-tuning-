# Demo card — one page, all numbers measured

Their challenge: *"An object itself usually isn't the anomaly. **The context is.**"*
Constraints: small model · real time · economical across many feeds.

---

## The one-sentence pitch

> Cheap noticing, expensive reasoning. A 5MB detector watches every frame; a
> arithmetic context layer decides which **0.7%** of frames deserve a VLM; and the
> VLM is asked only what a still frame can actually answer.

---

## Numbers to quote

| | 1080p (typical drone feed) | 4K (worst case) |
|---|---|---|
| Throughput | **55.1 fps** | 26.6 fps detector / 15.1 fps full |
| Per frame | **18.0 ms** | 40–61 ms |
| Real time? | **4.4x headroom** | yes, at stride 2 |
| **Feeds per GPU** | **4.41** | 1.19 |
| Frames reaching the VLM | **0.67%** | 0.19% |

Hardware: one **GTX 1650, 4GB**, no Tensor Cores. Not a datacentre card.

**Accuracy vs ground truth** (a real vehicle composited into a live lane, so the
anomaly frame and box are known exactly):

| | Day | Night |
|---|---|---|
| Detection rate | 1.0 | 1.0 |
| Best IoU | 0.949 | **0.971** |
| Alert latency | +5.12s (= the dwell threshold, so on time) | +5.12s |
| **False alerts before the anomaly existed** | **0** | **0** |

**Generalization — same code, zero changes:**
highway 4K → `stopped_vehicle`; pedestrian plaza 1080p → `crowd_density`.

**Agreement with a 27B teacher** (Groq qwen3.8-27b labelled the same frames):

| | Value |
|---|---|
| **Precision** | **1.00** — it never cries wolf |
| Recall | 0.60 |
| F1 | 0.75 |
| Accuracy | 0.857 (n=14) |
| Severity MAE | 0.124 |
| Confusion | tp=3 fp=0 fn=2 tn=9 |

Both misses are the same shape: *"truck moving at only 6 km/h in a live lane"*.
Our rule asks "stopped for N seconds"; the teacher also flags **abnormally
slow**. That is a scope difference, not an error - and a named next rule.
Precision 1.00 is the deliberate operating point: for an operator-facing system,
alert fatigue is the enemy.

An earlier measurement showed adding the 3B VLM to the decision changed **zero**
boolean verdicts versus rules alone (identical F1), improving only severity
calibration - at ~45,000 ms per event against 0 ms. That is the measured
justification for letting arithmetic own the decision.

> **Say the caveat before they ask:** n is 10-13 frames with 4 positives. F1 0.857
> has wide error bars - it is a sanity signal, not a benchmark. The numbers worth
> defending are **zero false positives across 538 frames** and a **0.67% VLM
> invocation rate**, both of which come from large samples.

**The most interesting row is the last one.** Adding a 3B VLM to the decision
changed **zero** boolean verdicts and cost 45s per event. It improved only
severity calibration (0.092 -> 0.078 MAE). That is the measured justification
for letting arithmetic own the decision.

Both configurations miss the same single frame: a vehicle at 0.063
body-lengths/sec that the teacher judged stopped and our threshold judged moving.
A genuinely borderline call, and exactly the kind of judgement distillation on
real labelled volume would absorb.

---

## The three things that make this different

**1. Context is arithmetic, not vibes.**
A stopped car alerts. A stopped car in a jam does not — we track the
stopped-neighbour ratio, so congestion is explained away rather than alerted on.
Speed is scale-invariant (body-lengths/sec), so altitude changes don't break it.

**2. We measured that a small VLM cannot do the boolean — and redesigned around it.**
A single still frame contains no motion; a moving car and a stopped car are
pixel-identical. qwen2.5vl:3b scored **3/6 (chance)** across four prompt
revisions, flipping all-true/all-false with prompt emphasis while its *prose*
stayed correct. So the tracker owns the boolean, and the VLM is asked only what
it can see — hazards and surroundings. It can **escalate** to 0.9 on visible
fire/smoke/collision/crowd, but can never silently clear a stop we measured.

**3. It knows when not to make a claim.** (three separate gates, all measured)
A vehicle is its own ruler (car ≈ 4.4m), giving metres-per-pixel with no training
and no camera intrinsics. Scale spread across the frame is a free obliquity
estimate — above 2.5 the geometry can't support a speed, so the km/h figure is
**withheld rather than asserted**. Our highway view scores 3.32 and correctly
suppresses a wrong "12 km/h".

The same principle runs through two more gates:
- **Depth motion.** A truck driving straight at the camera barely moves in the
  image, so it read as "stationary in a live lane". We now require stillness in
  depth too, via rate of change of apparent size. The threshold was set from
  measured distributions (parked median 0.009 vs approaching 0.101) after the
  instantaneous signal proved too noisy to separate them at all.
- **Object size.** Below 2% of the frame diagonal we decline to judge dwell,
  because a receding vehicle that small cannot be distinguished from a parked one.

Ask me about the false-positive hunt — raising recall 3x exposed four real bugs,
including one where our own eval was under-counting false positives. Fixing the
measurement came before fixing the number.

---

## Live demo order

1. `demo_annotated.mp4` — red box + alert banner on the stopped truck, green
   tracks with dwell timers on flowing traffic.
2. Run it on **their** footage live — one flag: `--source <path>`.
   Frame-folder benchmarks work too; `--night` for low light.
3. `src\eval.py` — read out detection rate, IoU, zero false positives.
4. Show `0.67%` trigger rate → that's the scaling argument.

## If asked "why not just use the VLM for everything?"

Because we measured it: 27–45s per call on this GPU, contending with the
detector for the same 4GB, and 3/6 accuracy on the judgement. The tracker answers
the same question in **0 ms** with 100% recall on ground truth. Using a VLM where
arithmetic suffices is how you fail the "economical across many feeds" constraint.

## Honest limitations (say these before they ask)

- Night testing is a luminance/noise simulation — no headlight glare or motion blur.
- The km/h estimate needs a near-nadir view; we detect and disclose when it doesn't hold.
- The distillation set is small; the pipeline is proven, the fine-tune is a
  demonstration until real labelled volume exists.
