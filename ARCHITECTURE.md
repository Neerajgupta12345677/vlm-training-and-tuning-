# Architecture — Drone Video Anomaly Detection

Real-time, context-dependent anomaly detection in aerial drone video on a
**4 GB GTX 1650**, using a small VLM sparingly enough to stay affordable.

The organising idea is that "noticing" and "reasoning" have wildly different
costs, so they must be different stages. Cheap per-frame work runs on every
frame; the expensive model is consulted on the **0.19–1.9%** of frames that
earn it.

---

## End-to-end flow

```
video file
   │
   ├─ STAGE 0  calibrate_zones.py                        once per video
   │     └─ zone polygons (carriageway / shoulder / parking)  → T###_zones.json
   │
   ├─ STAGE 1  detect_track.py                           every frame
   │     ├─ threaded decode (iter_frames_threaded)
   │     ├─ YOLO26n, --aerial (imgsz 1280, conf 0.10)
   │     │    NB: defaults to STOCK weights. The VisDrone fine-tune exists at
   │     │    models\yolo26n_visdrone.pt but is only used if --weights names it.
   │     ├─ ByteTrack (supervision) → persistent track ids
   │     └─ ego_motion.py: Shi-Tomasi + LK + RANSAC partial affine
   │           → track positions in a STABILISED reference frame
   │
   ├─ STAGE 2  context_state.py                          every frame, no model
   │     └─ per-track dwell, speed, zone, neighbour state
   │           → stopped_vehicle · slow_vehicle · collision_signature
   │             traffic_congestion · wrong_way_vehicle · person_in_roadway
   │             loiter · crowd_density
   │           → rule_anomalous / rule_severity        ← OWNS THE BOOLEAN
   │
   ├─ STAGE 2b appearance_classifier.py                  sampled frames
   │     └─ MobileNetV3-Small, appearance classes, threshold 0.72
   │           classify_video()      → clip label        (Level 1)
   │           windows_for_label()   → intervals         (Levels 2/3)
   │
   │     (STAGE 2c motion classifier was BUILT AND REJECTED - see below.
   │      It is not in the pipeline. Code kept: build_motion_frames.py,
   │      train_motion.py, motion_classifier.py.)
   │
   ├─ STAGE 3  vlm_reason.py                             event-triggered only
   │     └─ Qwen2.5-VL-3B via Ollama → SceneObservation
   │           (hazard_visible, hazard_type, surroundings)
   │           may ESCALATE, never silently overturn
   │
   └─ FUSION + EXPORT
         combine_predictions.py  class-priority fusion + stalled/blocking gate
         explain_events.py       scene-first explanations (reasoning bonus)
         attach_l23_times.py     adaptive-rate interval extraction
         export_arena.py         submission JSON + runtime_metadata
         score_arena.py          reproduces the live leaderboard locally
```

---

## Why the work is split this way

### A still frame does not contain motion

This is the single constraint the whole design bends around. A moving car and a
stopped car are **pixel-identical in one image**. Anything defined by duration
or direction — loitering, stalling, blocking, wrong-way, congestion — cannot be
read off a single frame, by any model, at any size.

Measured consequence (four prompt revisions, 3 positives / 3 negatives each):
Qwen2.5-VL-3B scored **3/6 every time — chance**, while its *prose* stayed
accurate ("the truck is stationary in a live driving lane, surrounding traffic
still flowing"). Its boolean contradicted its own description.

So responsibilities follow evidence type:

| decision | owner | why |
|---|---|---|
| is it anomalous? | **Stage 2 rules** | dwell, zone, neighbour state are arithmetic |
| what is visible? | **VLM** | description is what it is genuinely good at |
| appearance classes | appearance classifier | fire/smoke/flood/spill *are* visible in one frame |
| motion classes | *unsolved* | see "the motion classifier did not work", below |

The VLM may **escalate** (a visible hazard promotes an event to anomalous at
severity ≥ 0.9) but may never clear a stop the tracker measured. `--decision`
selects `hybrid` / `rules` / `vlm`; the last exists only to keep the gap
measurable.

### Ego-motion compensation is load-bearing

Every Stage 2 signal is computed from where an object sits in the **image**,
which equals where it sits in the **world** only if the camera is bolted down.
A drone pans and drifts, so a parked car has non-zero image velocity and the
stopped-vehicle rule silently never fires. Nothing errors — the system just
goes quiet. Partial affine (4 DOF) rather than full homography (8 DOF), because
it fits far more stably from sparse noisy correspondences.

### Why a separate motion classifier

`train_appearance.py` originally handled four motion classes on RGB frames. Its
own source comment flagged the risk:

> *wrong_way_driving is the weakest of the four on principle — one frame
> genuinely cannot show direction — so treat its val recall with suspicion: if
> it scores well it may be reading scene furniture rather than heading.*

The public leaderboard confirmed it. **Every** failing class was in that group,
and no appearance class was:

| class | found | false alarms |
|---|---|---|
| fighting_or_violence | 0 / 3 | 1 |
| vehicle_blocking_traffic | 0 / 2 | 4 |
| wrong_way_driving | 0 / 1 | 1 |
| loitering | 2 / 7 | 4 |

Those four produced **10 of 19 false alarms**.

The cause is structural, not a tuning error. Training clips are **trimmed to
the event** — median event coverage is 100% for 7 of 10 classes, and loitering
yields 810 in-event frames against **0** frames of the same scene with nothing
happening. So class correlates perfectly with background, background is the
cheapest available feature, and background is **constant within a video**. A
per-frame score that does not vary across a clip cannot localise anything, at
any threshold — which is exactly why Level 3 scored 20%. (Confirmed
independently: ranking windows by confidence put the wrong window first in all
4 L3 videos.)

**The fix is the input, not the thresholds.** `build_motion_frames.py` caches
ego-compensated temporal *differences* at three time-scales (0.4 / 1.5 / 4.0 s,
one per channel). Static scene content cancels to black, so background
memorisation is unavailable and the network is pushed onto motion — which
defines these classes and *does* vary within a clip. It also makes the 632
`normal` videos usable as negatives: they are a different scene, which made them
worthless against an appearance model, but once the scene is subtracted "people
moving through" vs "a person who stays put" is like-for-like.

Two input-specific training choices, both deliberate:
- **No horizontal flip** — mirroring reverses direction of travel, which *is*
  the label for wrong_way_driving.
- **No saturation jitter** — the channels are time-lags, not colours.

### …and the motion classifier did not work

It trained well — best val macro_recall **0.705**, loitering 27/27 on held-out
clips. It then failed both tests that matter, and was **rejected**.

**Test 1 - does it localise?** Fired-fraction vs the ground-truth anomalous
fraction on the Level-3 videos:

| video | GT anomalous | appearance fires | motion fires |
|---|---|---|---|
| T031 | 35% | 0% | 0% |
| T032 | 24% | 65% | **95%** |
| T033 | 19% | 65% | 47% |
| T034 | **2%** | 75% | **100%** |

Only T033 improved. T034 went from a 30x over-fire to firing on *every* sampled
frame.

**Test 2 - does it classify better?** Swapping the class label on motion-class
videos, timing untouched: **fixed 0, broke 1, NET -1**. It abstains on most
clips, and where it fires confidently it reproduces the appearance model's own
error (T021/T022: says loitering, truth is fighting).

**Why the reasoning was half right.** Differencing does remove the background,
and the network can no longer memorise the scene. But the premise that *motion*
varies within these clips is false. On a six-minute pedestrian scene the motion
character is as constant as the appearance - people move throughout. The 2% of
T034 that is labelled anomalous is not a stretch where the scene moves
differently; it is where **one particular person** has been dwelling too long.

That is the real lesson, and it generalises past this one model: **the evidence
is a property of a single track, not of the frame.** No whole-frame classifier
of any input type - RGB, difference, optical flow - can represent it, because
the quantity being measured does not exist at frame level. Level 3 needs
per-track temporal state, which only the tracker has. Stage 2 already computes
exactly that; the unfinished work is emitting intervals from it rather than
from a classifier's per-frame scores.

The code is kept (`build_motion_frames.py`, `train_motion.py`,
`motion_classifier.py`) because the negative result is worth more than the
silence - it rules out an entire family of approaches by measurement.

`stalled_or_broken_down_vehicle` is excluded from both classifiers: **4
training videos exist in the entire dataset**. That is a data ceiling, not a
training problem. It stays with Stage 2, whose stopped-vehicle rule produced
the only true positive of the rules-only run.

### The detector was the original weak link

Stock YOLO26n on real nadir drone footage (VisDrone val, 100 images, 5,189
objects):

| config | overall recall | tiny objects |
|---|---|---|
| stock — imgsz 640, conf 0.25 | 0.152 | 0.008 |
| **aerial — imgsz 1280, conf 0.10** | **0.413** | **0.149** |

Stock settings miss **85% of aerial objects**. Config alone nearly triples
recall for free, and imgsz 1536 still clears 25 fps. `--aerial` is now the
default.

The detector is additionally fine-tuned on **VisDrone** (Kaggle T4, 30 epochs,
imgsz 1024): 2.25× aerial recall, 7× on tiny objects. VisDrone also carries
`tricycle` / `awning-tricycle`, which COCO cannot represent at all — the
closest public proxy to auto-rickshaws.

> The competition dataset ships **event-level labels only — no bounding boxes**,
> so the detector cannot be fine-tuned on it directly. VisDrone is the correct
> substitute.

---

## Measured performance (4K 25 fps, GTX 1650)

| configuration | throughput |
|---|---|
| Stage 1 alone, threaded decode | 26.6 fps — 1.06× real-time on 4K |
| Stage 1, serial decode | 20.0 fps (decode 23.5 ms + detect 29.3 ms serialise) |
| full pipeline, `--decision rules`, stride 2 | **14.9 fps vs 12.5 needed — real-time** |
| `--decision hybrid` (VLM in loop) | 6.8 fps — not real-time on one card |
| annotated 4K encode | 26.6 → 9.7 fps (hence `--save-width 1280`) |

VLM warm latency is **27–45 s per call**, and it contends with YOLO for the
same 4 GB. **One 4 GB GPU cannot run a real-time detector and a 3B VLM
concurrently** — `rules` for a strict real-time demo, `hybrid` for enrichment
or a second GPU.

---

## Where training happens

| target | where | why |
|---|---|---|
| YOLO26n → VisDrone | Kaggle T4 | needs hours of GPU |
| Qwen2.5-VL-3B QLoRA | Kaggle T4 | 4 GB VRAM cannot hold weights + gradients + optimizer state |
| MobileNetV3-Small (appearance) | local GTX 1650 | small enough |
| MobileNetV3-Small (motion) | local GTX 1650 | small enough |
| **all inference** | local | Ollama holds VLM weights in its own process |

Use the **T4, never the P100** — current bitsandbytes/Unsloth 4-bit kernels are
no longer built for sm_60 and the run dies at `get_peft_model`.

---

## Fine-tuned VLM adapter status

`C:\dvad\models\hf_qwen25vl_ahc_lora` — LoRA on Qwen2.5-VL-3B-Instruct, r=16,
alpha=16, **language layers only** (vision tower untouched), macro-F1 0.437
with multi-frame aggregation.

**MERGED AND LIVE** as the Ollama model `qwen25vl-ahc` (3.3 GB).

An HF/PEFT adapter cannot be applied to a GGUF at runtime, so it had to go
fp16 base → apply LoRA → merge → GGUF → quantize Q4_K_M → `ollama create`. That
ran on a Kaggle T4, because the merge alone needs ~6.2 GB RAM against 7.3 GB
total on this laptop, and the fp16 base is a 7.5 GB download. Only the finished
3.27 GB (model + vision projector) came back.

Verified rather than assumed: the notebook asserts the adapter actually
attaches before exporting, and logged **504 LoRA modules injected**
(`model.layers.*.self_attn.q_proj.lora_A`, …). Without that check a module-name
mismatch would have produced a silent no-op merge and shipped the stock model
as "fine-tuned". The quantizer's 5886 MiB → 1834 MiB confirms it processed real
merged weights.

Self-test result on the stopped-truck frame: `anomalous: true, severity 0.7` —
*"A truck has stopped in a live traffic lane while other vehicles are still
flowing around it."* Stock `qwen2.5vl:3b` scored **3/6 (chance)** on this same
judgement across four prompt revisions, with its boolean contradicting its own
prose.

Two caveats that keep this honest:
- Measured latency was **118 s** on a cold call. Warm calls are faster (stock
  measured 27–45 s), but this does not become real-time on a 4 GB card.
- The scored 50.7 run used `--decision rules` and **never called the VLM at
  all** — every `model_runtimes` entry reads `appearance-classifier`. The merge
  makes the fine-tune available for `hybrid` enrichment; it does not by itself
  change D1/D2/D3.

Because the adapter touches only language layers, the **mmproj vision projector
transfers unchanged**.

---

## Known gaps (stated, not hidden)

1. **Level 3 is 0-for-8.** Timing must come from per-track state, not per-frame
   classifier scores. This is the single largest pool of unclaimed marks (32).
2. **The VisDrone detector fine-tune is not wired in by default.** `--weights`
   defaults to `None`, so runs use stock COCO YOLO26n. Passing
   `--weights C:\dvad\models\yolo26n_visdrone.pt` should be the default.
3. **`traffic_congestion` (23 videos) and `stalled_or_broken_down_vehicle` (4
   videos)** are data-limited, not model-limited.
