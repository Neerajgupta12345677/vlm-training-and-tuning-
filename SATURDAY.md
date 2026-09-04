# Saturday runbook — Sept 5, FlytBase Labs

**Event:** Visual Intelligence Hackathon, 9:00–19:00, FlytBase Labs, Baner, Pune.
Build window is **11:00–18:00 (7 hours)**; demos 18:00–19:00.
Problem statement, evaluation criteria and submission format are revealed on the day.
Organisers supply **real urban drone footage including night conditions** plus
pre-prepared public benchmark datasets.

Their stated challenge is *"An object itself usually isn't the anomaly. The context is."*
with three constraints — small model, real time, economical across many feeds.
Those map onto numbers this pipeline already prints:

| Their constraint | Our evidence | Where |
|---|---|---|
| Context, not objects | Stage 2 dwell + zone + stopped-neighbour ratio; a lone stop alerts, a jam does not | `context_state.py` |
| Small model | YOLO26n (5.3MB) + a 1.8–3B VLM, all local on a 4GB GTX 1650 | run header |
| Real time | 55.1 fps at 1080p, 4.3x the rate needed | run summary |
| Economical | **4.41 feeds/GPU at 1080p**; only **0.67% of frames** reach the VLM | run summary |

**Both night footage and frame-folder datasets are already tested — see §1b and §1c.**

Everything below is copy-pasteable. `PY` is the venv interpreter; nothing uses
bare `python` (the system default is 3.13t freethreaded and has no ML wheels).

```
set PY=C:\dvad\.venv\Scripts\python.exe
```

---

## 0. Sanity check (run this FIRST, before the dataset arrives)

```
%PY% -c "import torch; print('cuda', torch.cuda.is_available())"
%PY% src\context_state.py --selftest
%PY% src\vlm_reason.py --selftest --backend mock
```

All three must pass. The third needs no weights and no network — if that works,
you can demo even with the wifi dead.

---

## 1. The real dataset just dropped

Put the videos anywhere, e.g. `C:\dvad\data\real\`. Then, per video:

```
:: (a) derive zones from observed motion - no hand-drawing, ~40s on 4K
%PY% src\calibrate_zones.py --auto --source C:\dvad\data\real\clip1.mp4

:: (b) LOOK AT THE PREVIEW before trusting it
::     opens C:\dvad\data\real\clip1_zones.jpg - lanes should follow the road
::     and the flow arrows should point the way traffic actually travels

:: (c) run the pipeline
%PY% src\pipeline.py --source C:\dvad\data\real\clip1.mp4 ^
  --zones C:\dvad\data\real\clip1_zones.json ^
  --decision rules --stop-seconds 20 --stride 2 ^
  --save C:\dvad\outputs\clip1_annotated.mp4
```

If auto-calibration produces nonsense (unusual camera angle, static scene, no
vehicle motion to learn from), fall back in this order:

```
%PY% src\calibrate_zones.py --draw --source ...        :: draw by hand, 1-2 min
%PY% src\calibrate_zones.py --whole-frame --source ... :: instant, crude
```

Or just omit `--zones` entirely — Stage 2 treats unmapped areas as lane-like
and still triggers, it simply loses the parking/shoulder distinction.

### Whole folder at once
```
%PY% src\pipeline.py --data_dir C:\dvad\data\real --limit-videos 0 --decision rules
```

## 1a-CRITICAL. Drone footage — ALWAYS add `--aerial`

```
%PY% src\pipeline.py --source <drone_clip> --zones <zones> --decision rules --aerial
```

This is the single highest-impact flag in the system. Measured on VisDrone
(real aerial imagery, 5189 ground-truth objects):

| config | overall recall | small objects |
|---|---|---|
| default (imgsz 640, conf 0.25) | **0.152** | **0.008** |
| `--aerial` (imgsz 1280, conf 0.10) | **0.413** | 0.149 |
| imgsz 1536, conf 0.10 | **0.462** | 0.204 |

**Default settings miss 85% of aerial objects and are blind to small ones.** Our
own highway test clip was flattering because its vehicles were large and side-on.

It does not cost real time: 41.7 fps with `--aerial` on 1080p, 3.33 feeds/GPU.
For very high altitude, push further: `--imgsz 1536 --conf 0.1` (31.5 fps).

**If the pipeline looks broken on their footage, this is the first thing to try —
the fault will be Stage 1 detection, not the anomaly logic.** Sanity-check what
the detector actually sees before touching thresholds:
```
%PY% src\detect_track.py --source <clip> --imgsz 1280 --conf 0.1 --save C:\dvad\outputs\check.mp4
```

## 1b. Night footage — add `--night`

```
%PY% src\pipeline.py --source <night_clip> --zones <zones> --decision rules --night
```

`--night` only lowers the detector confidence to 0.20. That is the whole fix, and
it was chosen by measurement:

- Night raw already keeps **90%** of the daylight detection count.
- conf 0.20 recovers detections to **above** the daylight baseline (581 vs 530).
- **Do NOT reach for CLAHE.** Measured on 4K it cost **23.7 → 4.0 fps** and found
  *fewer* objects (468 vs 479). Histogram equalisation is a trap here.
- End-to-end on simulated night: anomaly still caught at the identical timestamps,
  IoU **0.971** (better than daylight), 0 false positives, 22.2 fps.

Caveat: our night test is a luminance/noise simulation. Real night footage adds
headlight glare and motion blur. If detection collapses on their real night clips,
the first lever is still `--conf` (try 0.15), then `--imgsz 960`.

## 1c. Benchmark datasets that ship as frame folders

UCSD Ped, CUHK Avenue, ShanghaiTech and UCF-Crime ship numbered frames, not videos.
Point `--source` at the folder — it is handled natively:

```
%PY% src\pipeline.py --source C:\dvad\data\benchmark\clip01 --decision rules
```

Frame folders carry no frame rate, so 25fps is assumed. **If the dataset states a
different rate, set it** — every dwell threshold depends on it:

```
set DVAD_FRAME_SEQ_FPS=10
```

---

## 1d. What the system can flag (all class-agnostic, all from Stage 2)

| Rule | Fires when | Rule verdict |
|---|---|---|
| `stopped_vehicle` | vehicle stationary > `--stop-seconds` in a live lane | anomalous, 0.55–0.85 (scales with dwell) |
| ...but congested | most other vehicles are stopped too | **benign**, 0.2 — a jam is not an incident |
| `person_in_roadway` | person in a driving lane | anomalous, 0.85 |
| `loitering` | person stationary > `--loiter-seconds` (default 25) | 0.7 in a lane/restricted area, else 0.35 |
| `wrong_way_vehicle` | heading deviates > `--wrong-way-tolerance` deg from calibrated flow | anomalous, 0.9 |
| `crowd_density` | more live person tracks than `--crowd-count` (default 8) | benign 0.3 — the VLM decides if it's a queue or a problem |
| `slow_vehicle` **(off by default)** | vehicle crawling vs the median of its moving neighbours, needs `--enable-slow-vehicle` | anomalous, 0.3–0.55 |

**On `slow_vehicle`:** it works, but it is off by default on purpose — its
thresholds were tuned against a single oblique clip and it costs 1 false
positive on the aerial ground-truth run. Zero-false-positives is the strongest
claim this system has; don't trade it for a rule you can't validate. If their
footage has obvious crawling-vehicle anomalies, turn it on and **re-check the
false positive count before demoing it**.

The VLM can **escalate** any of these to 0.9 if it sees fire, smoke, a collision,
debris or a crowd forming — matched against a fixed allow-list in
`vlm_reason._is_real_hazard()`, not accepted from free-form model output. That
list is deliberate: moondream once reported `hazard_type: "person"` for an
ordinary crowd scene, which a looser check ("anything but 'none'") would have
escalated to a false alert. It can never silently clear a stop the tracker
measured.

All three thresholds (`--loiter-seconds`, `--crowd-count`,
`--wrong-way-tolerance`) and all six rules have been run against real footage
with real detections, not just the synthetic selftest — see PROGRESS.md
"RULE COVERAGE CLOSED" for the positive/negative control results.

## 1e. Real-world distances (the car-length ruler)

A detected vehicle is its own ruler — a car is ~4.4m, a truck ~8m, so
metres-per-pixel comes straight off the box. No training, no camera intrinsics.
Calibrated per-track, so perspective cancels itself.

It also **refuses to guess**. Scale spread across the frame gives a free
obliquity estimate; above `max_obliquity_for_speed` (2.5) the km/h figure is
computed but withheld, because in an oblique view vehicles move along the depth
axis and image-plane motion understates their real speed. Measured: the highway
bridge view scores 3.32 and correctly suppresses a wrong 12 km/h reading.

If Saturday's footage is near-nadir drone video, expect real km/h in the context
string — check `features.traffic_flow_kmh_reliable` in the events jsonl.

## 2. Tuning knobs, in the order you'll reach for them

| Symptom | Fix |
|---|---|
| No alerts at all | lower `--stop-seconds` (try 8, then 5) |
| Too many alerts | raise `--stop-seconds`, raise `--cooldown` |
| Missing small/distant objects | `--conf 0.2`, or `--imgsz 960` |
| Too slow / not real-time | `--stride 3`, or `--decision rules` |
| Parked cars flagged as anomalies | calibrate zones so parking is labelled |
| Everything reads "unmapped" | zones don't cover the road — re-run `--auto`, or raise `--dilate` |

---

## 2b. ONE COMMAND for the demo

Don't assemble flags in front of judges. This calibrates zones, runs the
pipeline, writes the annotated video, and prints a readable results block:

```
%PY% src\demo.py --source <their_clip>
%PY% src\demo.py --source <their_clip> --night
%PY% src\demo.py --source <their_clip> --quick        :: first 300 frames
```

**Quote throughput from a `--no-video` run.** Encoding the annotated video costs
more than the pipeline itself, so the default run understates you badly —
measured on the 4K clip: 13.3 fps / 1.06 feeds-per-GPU with encoding vs
**20.1 fps / 1.61** without. Show the video from the first run, quote the
numbers from this one:
```
%PY% src\demo.py --source <their_clip> --no-video
```

Every step degrades gracefully — if zone calibration fails, the run still
proceeds (unmapped areas are treated as lane-like and the rules still fire).

## 3. Demo script (what to actually show)

1. **The annotated video.** Red box + alert banner on the stopped vehicle,
   green tracked boxes with dwell timers on everything else.
2. **The cost argument.** From the run summary: Stage 3 fires on **~0.2% of
   frames**. Cheap noticing, expensive reasoning only when earned. This is the
   scaling story.
3. **The eval numbers**, against a composited ground-truth anomaly:
   ```
   %PY% src\eval.py --ground-truth C:\dvad\data\vehicles_stopped_ground_truth.json ^
     --predictions C:\dvad\outputs\events_rules.jsonl ^
     --run-summary C:\dvad\outputs\summary_rules.json
   ```
   detection rate 1.0 · IoU 0.949 · +5.1s latency (= the dwell threshold, so
   exactly on time) · **0 false alerts before the anomaly existed**.
4. **The honest engineering finding** — this is a strength, not an apology.
   We measured that a 3B VLM cannot reliably decide "is this anomalous" from a
   single frame, because *a still frame contains no motion*. It scored at chance
   across four prompt revisions while describing the scene correctly. So the
   deterministic tracker owns the boolean and the VLM does what it is genuinely
   good at: seeing hazards and explaining the scene. Details in CLAUDE.md.

---

## 4. If the wifi dies

Nothing in the demo path needs the network - and this was actually tested, not
just reasoned about: with all external traffic routed to an unroutable address
(a real network blackhole, external DNS/HTTP genuinely unreachable), both
`--decision rules --aerial` (24.8s, correct detection) and `--decision hybrid
--backend ollama` (real 15.9s local inference call) completed correctly.
- YOLO weights are cached in `C:\dvad\models\`
- Ollama serves locally on `localhost:11434` - loopback traffic, never touches
  the network hardware, works identically whether wifi is up or down
- `--decision rules` needs no model at all
- `--backend mock` runs the whole pipeline with zero weights
- Ultralytics telemetry (`sync`) is disabled - one less thing that could try
  to phone home and stall

Only the *teacher labelling* and *Kaggle upload* need internet, and those are
offline-prep steps, not demo steps.

---

## 5. Training half

One-time setup:
```
:: Groq gives a free key with 14,400 requests/day - console.groq.com/keys
setx GROQ_API_KEY "gsk_..."          :: then open a NEW terminal
```

**Kaggle is already set up and verified** — authenticated as `guptaneeraj123`,
upload and versioning both tested against a real private dataset. Nothing to do.
If it ever needs redoing (Kaggle now uses a standalone `KGAT_...` token, not
kaggle.json):
```
%PY% src\setup_kaggle.py --token KGAT_xxxxxxxx
%PY% src\setup_kaggle.py --verify-only
```

Then:
```
:: harvest candidates (no model needed, works offline)
%PY% src\pipeline.py --source <clip> --zones <zones> --no-vlm ^
  --stop-seconds 5 --cooldown 4 --sample-normal 10 --stride 2 ^
  --out C:\dvad\outputs\harvest_events.jsonl

:: check the plan first, ALWAYS
%PY% src\distill_label.py --events C:\dvad\outputs\harvest_events.jsonl --dry-run

:: label with the free Groq teacher (Llama 4 Scout, vision)
%PY% src\distill_label.py --events C:\dvad\outputs\harvest_events.jsonl --limit 40

:: if that model id has drifted, ask the provider what exists
%PY% src\distill_label.py --provider groq --list-models

:: package + upload
%PY% src\build_kaggle_dataset.py --labels C:\dvad\data\pseudo_labels.jsonl --push
```

Provider fallbacks: `--provider openrouter` (free but only **50 requests/day** on a
zero balance) or `--provider anthropic` (paid, best teacher quality).

Then run the fine-tune from the CLI — no browser clicks:
```
%PY% src\push_notebook.py --push --slug dvad-finetune-qwen2-5-vl --title "dvad finetune qwen2 5 vl"
%PY% src\push_notebook.py --wait  --slug dvad-finetune-qwen2-5-vl
%PY% src\push_notebook.py --pull  --slug dvad-finetune-qwen2-5-vl
```

Two Kaggle traps already hit and fixed — do not re-introduce them:
- **The URL slug comes from the TITLE, not the id.** "DVAD finetune Qwen2.5-VL"
  became `dvad-finetune-qwen2-5-vl`, so `--status` 404'd on a kernel that existed.
  Keep `--slug` and `--title` consistent.
- **nbformat needs a trailing `\n` on every source line.** Without it Kaggle joins
  the cell onto ONE line: a cell starting with `#` silently becomes a comment and
  does nothing, and a code cell dies with "SyntaxError: incomplete input".
  `notebooks/build_notebook.py` regenerates the notebook and asserts this.

If you edit the notebook, regenerate it rather than hand-editing the JSON:
```
%PY% notebooks\build_notebook.py
```
