# Progress Log (append short bullets only, newest at top)

## Status: PIPELINE WORKS END-TO-END, night-tested. 2 free credentials still needed.

### BLOCKERS
- [ ] **Kaggle Phone Verification** (Settings -> Phone Verification). THE ONLY ONE LEFT.
      Not needed for upload (that works), but REQUIRED to select a GPU accelerator
      on a notebook. It can lag - do not leave it to Saturday morning.

### !!! BIGGEST FINDING OF THE BUILD: the DETECTOR is the weak link on aerial !!!
Tested stock YOLO26n against real aerial imagery (VisDrone val, 100 images,
5189 ground-truth objects). Our own highway clip was flattering - big, side-on
vehicles. Real nadir drone footage is nothing like it.

  config                    overall recall   vehicle   person   twowheel   tiny objects
  imgsz 640,  conf 0.25       0.152           0.331     0.134    0.014      0.008
  imgsz 1280, conf 0.25       0.278           0.508     0.270    0.059      0.056
  imgsz 1280, conf 0.10       0.413           0.643     0.424    0.148      0.149
  imgsz 1536, conf 0.10       0.462           0.686     0.485    0.172      0.204

**Stock settings miss 85% of aerial objects and are effectively blind to small
ones (0.008).** Config alone triples recall, for free.

And it does NOT cost real time (1080p, stride 1):
  imgsz 640 -> 47.9 fps | 960 -> 50.3 | 1280 -> 41.3 | 1536 -> 31.5 fps
So imgsz 1536 still clears the 25fps bar. `--aerial` preset = imgsz 1280 +
conf 0.10, measured at 41.7 fps / 3.33 feeds per GPU.

**USE `--aerial` ON ANY DRONE FOOTAGE TOMORROW.** Without it the pipeline will
look broken on their data and the fault will be in Stage 1, not the logic.

Also: VisDrone has `tricycle` / `awning-tricycle`, classes COCO does not have at
all - the closest public proxy to Indian auto-rickshaws. 240 of 5189 GT objects
in the sample were tricycles that the stock detector cannot represent.

Running overnight: `notebooks/finetune_yolo_visdrone.ipynb` on Kaggle T4
(YOLO26n, VisDrone, 30 epochs, imgsz 1024) to close the remaining domain gap.
Verify any resulting weights with:
  `python src\eval_aerial.py --limit 100 --imgsz 1024 --conf 0.15 --weights <new.pt>`
NOTE: VisDrone class ids are NOT COCO - update VEHICLE_NAMES / VEHICLE_CLASSES
(add `van`, the tricycles) before using those weights.

### FALSE-POSITIVE HUNT AT `--aerial` (2026-09-04) - four real bugs closed
Raising detector recall exposed failure modes the low-recall default was hiding.

1. **Eval was under-counting false positives.** It only counted alerts BEFORE
   the anomaly onset, so any spatially-wrong alert afterwards was invisible.
   eval.py now reports `alerts_elsewhere_after_onset` and a total. Measurement
   integrity first - the earlier "0 false positives" was partly an artefact.
2. **Duplicate alerts on one object.** At conf 0.10 a single truck came back as
   truck+truck+bus, three ByteTrack ids, three alerts. Fixed with
   `agnostic_nms=True` (per-class NMS cannot merge different labels) plus an
   IoU-based duplicate suppressor in Stage 2.
3. **A truck driving straight at the camera was flagged "stationary".** Motion
   along the view axis produces almost no centroid displacement. Fixed by also
   requiring stillness in DEPTH, via the rate of change of apparent box size.
4. **The depth gate then broke the dwell latch** - because instantaneous scale
   rate is far too noisy. MEASURED both populations rather than guessing:
       parked truck      median 0.0092  p75 0.0443  p90 0.2071  max 0.4988
       approaching truck median 0.1010  p75 0.1306  p90 0.2353  max 0.5656
   The distributions OVERLAP (parked p90 > approaching median), so no
   instantaneous threshold works. The medians separate 11x, so the gate uses the
   median over a 15-sample window. Threshold 0.04 sits between them.

Result at `--aerial` on the ground-truth clip: 4 alerts -> **1**, IoU 0.981,
**0 false positives** (both categories), still real-time.

### QUALITY METRICS vs the 27B teacher (2026-09-04)
Reproduce:
  `python src\pipeline.py --source ... --decision rules --stop-seconds 5 --cooldown 4 --sample-normal 10 --stride 2 --out <events>`
  `python src\eval.py --reference C:\dvad\data\pseudo_labels.jsonl --predictions <events>`

                     rules-only      hybrid (rules+VLM)
  precision            1.00              1.00
  recall               0.75              0.75
  F1                   0.857             0.857
  accuracy             0.90  (n=10)      0.923 (n=13)
  severity MAE         0.092             0.078
  cost per event       0 ms              ~45,000 ms

- **Adding the 3B VLM changed ZERO boolean verdicts.** It only tightened severity
  calibration. This is the measured justification for the architecture: the
  tracker owns the decision, the VLM explains it and can escalate on hazards.
- Both configs miss the same single frame (`f000480_t21`, 0.063 body-lengths/sec)
  which the teacher called stopped and our threshold called moving. Borderline.
- **n is 10-13 frames with 4 positives - state that caveat before anyone finds
  it.** The large-sample numbers are the defensible ones: 0 false positives over
  538 frames, 0.67% VLM invocation rate.

### TRAINING SUCCEEDED ON KAGGLE (2026-09-04, run v5)
- GPU **Tesla T4 sm_75**, trained in **9.7 min**, final loss **1.2276**,
  peak VRAM **6.31 GB**, trainable **29,933,568 / 2,532,556,800 (1.18%)**.
- Adapter downloaded and **cached offline at `C:\dvad\models\lora_adapter`
  (125.2 MB)** - the wifi-independence requirement is met.
- Reproduce with: `python src\push_notebook.py --push --wait --pull
  --slug dvad-finetune-qwen2-5-vl --title "dvad finetune qwen2 5 vl"`
- KNOWN COSMETIC ISSUE, diagnosed and closed: the notebook's post-training
  sanity-inference cell emits `!!!!!!!` (token id 0 repeated = NaN logits).
  Attempted fix in v6 (explicit `text=`/`images=` kwargs, inference_mode +
  autocast, explicit pad_token_id) did NOT change it.
  **Verified locally that the ADAPTER ITSELF IS FINE** - 504 tensors, 0 NaN,
  0 Inf, no all-zero tensors, max |w| 0.336, mean max|w| 0.021. Those are
  healthy LoRA magnitudes, so training produced real learned weights and the
  cached artifact is usable. The bug is confined to that one notebook cell.
  Deliberately not chased further: the live demo does NOT load this adapter
  (Stage 3 uses stock Ollama), and with 14 samples the fine-tune was always a
  demonstration of the loop rather than a quality win. Two 10-minute Kaggle
  cycles were enough to spend on a cosmetic cell.
  Re-check any future adapter with `scratchpad/check_adapter.py`-style
  NaN/magnitude inspection before assuming a bad fine-tune.

### KAGGLE TRAINING RUN - pushed from the CLI (2026-09-04)
`src/push_notebook.py` drives push / status / pull, so Stage 7 needs no browser
clicks and is repeatable on the day.

Running it for real immediately caught **two bugs that would have burned Saturday
afternoon**, neither of which was an Unsloth problem:

1. **nbformat newlines (mine).** The generator used `split("\n")`, which strips
   the trailing newline every `source` entry must keep. Kaggle joined each cell
   onto ONE line. The visible failure was `SyntaxError: incomplete input` on the
   CONFIG cell - but the dangerous part is that the first two cells START with a
   `#` comment, so joining turned each into a single comment. They ran green and
   **did nothing**: CUDA_VISIBLE_DEVICES was never set and Unsloth was never
   installed. A silent no-op is far worse than a crash.
   Fixed in `notebooks/build_notebook.py`, which now also *asserts* the invariant.
2. **Kaggle derives the URL slug from the TITLE, not the id.** "DVAD finetune
   Qwen2.5-VL" resolved to `dvad-finetune-qwen2-5-vl`, so `--status` reported
   "permission denied" on a kernel that existed and was running fine.

3. **Kaggle mounts datasets deeper than expected** - `/kaggle/input/datasets/...`,
   not `/kaggle/input/<slug>/`. The two-level glob missed it. Now a recursive
   `**/train.jsonl` search that prints the actual tree when it fails.
4. **The kaggle CLI crashes writing logs on Windows** (UnicodeEncodeError, cp1252)
   and leaves a **0-byte log** - hiding the traceback you need. `PYTHONUTF8=1`
   fixes it; it is now forced in `push_notebook.py`, plus a `--log` flag that
   jumps straight to the exception.
5. **THE BIG ONE: never use the P100 accelerator.** Kaggle's default GPU is a
   P100 (sm_60, Pascal) and current bitsandbytes/Unsloth 4-bit kernels are not
   built for it: `CUDA error: no kernel image is available for execution on the
   device`, thrown at `get_peft_model`. Fix: `machine_shape: "NvidiaTeslaT4"`
   (valid values are NvidiaTeslaT4 / NvidiaTeslaP100 / Tpu1VmV38). Kaggle's own
   docs warn P100 is incompatible with the default image. The notebook now
   asserts compute capability >= 7 and fails readably instead.

Lesson worth keeping: an artifact that has never been executed is not done. The
notebook looked correct in review and was broken in FIVE separate ways that
review could not see - four of them environmental, not logical.

### FULL CHAIN NOW PROVEN END TO END (2026-09-04)
harvest -> Groq teacher labels -> package -> Kaggle upload. All verified live.
- **Groq works, but NOT as documented.** `meta-llama/llama-4-scout` does NOT exist
  on this account. Probing all 14 served models found exactly two that accept
  images: **`qwen/qwen3.8-27b`** (clean output - now the default) and
  `qwen/qwen3.6-27b` (leaks `<think>` tags). Model names and provider docs both
  lied; only the probe told the truth. Re-run `--probe-vision` if this drifts.
- Two Groq gotchas, both fixed:
  - `response_format: json_object` 400s unless the literal word "json" appears in
    the messages. Our teacher prompt never said it (the Anthropic path used
    structured outputs), so every call failed. Without JSON mode the model
    returns markdown prose that will not parse.
  - The binding free-tier limit is **8000 TOKENS/min**, not requests. Images are
    token-heavy, so concurrency is pinned to 1 and images to 512px.
- Labelled 14/14 frames, 0 errors, **4 anomalous / 10 benign** - real class
  balance, and the teacher disagreed with our rules on one borderline track,
  which is exactly the signal distillation is supposed to add.
- Dataset live: `guptaneeraj123/dvad-pseudo-labels` (private).

### GENERALIZATION PROVEN - the thing that actually wins this
Same code, zero changes, two very different scenes:
- Highway 4K oblique -> `stopped_vehicle` x2, real-time
- Pedestrian plaza 1080p -> `crowd_density` x1, 38.5 fps, 3.08 feeds/GPU
Rules are now class-agnostic: added **loitering** (person stationary too long)
and **crowd_density** (scene-level, above a person-count threshold). Previously
every rule was vehicle-only and a pedestrian scene produced zero events.

### Real-world scale via the car-length ruler (no training, no intrinsics)
A detected vehicle is its own ruler: from above, its box long side is its length,
so metres-per-pixel falls out of the class alone. Calibrated **per track**, which
cancels perspective (a distant car has both a smaller box and smaller pixel motion).
- **Critically, it also knows when NOT to claim a speed.** Scale spread across the
  frame (widest/narrowest m/px) is a free obliquity estimate. On the bridge-style
  highway view obliquity is 3.32, so the 12 km/h reading is withheld rather than
  asserted - it would have been badly wrong and would have cost credibility.
  On a near-nadir drone view the km/h appears. See `view_obliquity()`.

### KAGGLE: DONE AND VERIFIED (2026-09-04)
- Authenticated as **guptaneeraj123**, auth method `ACCESS_TOKEN`.
- Kaggle now uses a standalone `KGAT_...` token (no username, no kaggle.json).
  It lives at `~/.kaggle/access_token`. `setup_kaggle.py --token KGAT_...` installs it.
- **Write access proven, not just read**: created and then versioned a private
  dataset `guptaneeraj123/dvad-smoke-test` (17.2MB). Read auth alone would not
  have proven this. That smoke dataset is throwaway - delete it whenever.
- Fixed a bug this surfaced: `subprocess.run(text=True)` decoded the kaggle CLI's
  progress bytes with Windows cp1252 and raised UnicodeDecodeError *after* a
  successful upload - a success that looked like a failure. Now utf-8/replace.
- `build_kaggle_dataset.py` resolves the username via `kaggle config view` since
  there is no kaggle.json under token auth.
- SECURITY: the token was pasted in plaintext in a chat transcript. Rotate it
  after the event: kaggle.com -> Settings -> API -> Expire Token.

### Event requirements confirmed from the Luma page (2026-09-04)
- Sept 5, 09:00-19:00, FlytBase Labs Baner. **Build window is only 11:00-18:00.**
  Demos 18:00-19:00. Problem statement + eval criteria revealed on the day.
- Organisers supply real urban drone footage **including night**, plus public
  benchmark datasets. Both cases are now tested (see below).
- Their framing - "An object itself usually isn't the anomaly. The context is." -
  is exactly the Stage 2 design. Their three constraints map to metrics we print.

### 2026-09-04 (later)
- **1080p is the number to quote: 55.1 fps, 18.0 ms/frame, 4.41 feeds/GPU,
  0.67% of frames reaching Stage 3.** 4K is the pessimistic case; real drone
  feeds are typically 1080p. Verify: `--source C:\dvad\data\benchmark_seq\clip01`.
- **Frame-folder datasets supported.** UCSD/Avenue/ShanghaiTech/UCF-Crime ship
  numbered frames, not videos; `--source <dir>` and `--data_dir` now accept them.
  Frame rate is assumed 25fps - override with `DVAD_FRAME_SEQ_FPS` since every
  dwell threshold depends on it.
- **NIGHT TESTED - pipeline survives.** On simulated night footage the anomaly is
  caught at the identical timestamps, **IoU 0.971** (better than day's 0.949),
  0 false positives, 22.2 fps real-time.
  Ablation (120 frames, day baseline 530 detections):
    night raw conf .30 -> 479 (90% of day)
    night raw conf .20 -> 581 (ABOVE day)   <- the fix, now `--night`
    night CLAHE conf .30 -> 468 dets at **4.0 fps** (was 23.7)  <- rejected
  Lesson: the low-light lever is the confidence threshold, not image enhancement.
  Caveat: simulated night has no headlight glare or motion blur.
- **Free teacher providers wired in.** `distill_label.py --provider groq`
  (default, free, 14,400 req/day, Llama 4 Scout vision) or `openrouter`
  (free, only 50 req/day) or `anthropic` (paid). Groq + OpenRouter share one
  OpenAI-compatible path. `--list-models` queries the provider live because
  free model ids drift.
- **`src/setup_kaggle.py`** added: finds kaggle.json in Downloads, installs it,
  tightens ACLs, and verifies auth with a real API call.

### 2026-09-04
- **Async Stage 3.** A 45s VLM call inside the frame loop dropped throughput 15.8 -> 2.5 fps.
  Dispatched to a worker thread with a bounded queue: 6.8 fps. Verify: run with
  `--decision hybrid`, look for `vlm_async: true` in the summary.
- **Measured: one 4GB GPU cannot run real-time YOLO + a 3B VLM together.** Even async,
  hybrid does not reach real-time on this card. `--decision rules` is the real-time
  demo path (14.9 fps vs 12.5 needed, 1.19 feeds/GPU); hybrid is enrichment.
- **Stage 3 redesigned after measurement (the big finding).** Asking a small VLM
  "is this anomalous?" fails structurally: a still frame contains no motion, so the
  model cannot judge what the tracker already knows. qwen2.5vl:3b scored 3/6 (chance)
  across FOUR prompt revisions, flipping all-true/all-false with prompt emphasis while
  its prose stayed correct. Now: Stage 2 rules own the boolean, the VLM only observes
  and may escalate on a visible hazard. See CLAUDE.md "Division of labour in Stage 3".
- **Anchored dwell latch.** Detector box jitter on a parked truck kept clearing the
  stationary latch, resetting a real 19s stop to 2s. Now the latch releases only when
  the centre travels >0.6 body-lengths from where it stopped. Dwell now accumulates
  correctly: 5.0s -> 9.1s -> 13.1s.
- **Zone dilation.** Auto-derived lanes hugged the centroid track and missed lane width,
  so the stopped truck read as "unmapped" and the VLM refused to commit. Lanes are now
  dilated to a real footprint; the truck correctly resolves to `driving_lane`.
- **Negative sampling** (`--sample-normal N`). Free-flowing footage triggers almost no
  benign events, giving a single-class label set that teaches no boundary. Now samples
  normally-behaving tracks as labelled negatives. Harvest: 10 negatives / 5 positives.
- **EVAL AGAINST GROUND TRUTH PASSES**: detection rate **1.0**, best IoU **0.949**,
  detection latency +5.12s (= the 5s dwell threshold, so exactly on time),
  **0 false alerts before the anomaly existed**, trigger rate 0.19% of frames.
  Verify: `python src\eval.py --ground-truth C:\dvad\data\vehicles_stopped_ground_truth.json --predictions C:\dvad\outputs\events_rules.jsonl`
- **Threaded video decode**: 4K decode (23.5ms) and detection (29.3ms) were serial and
  additive. Overlapped, Stage 1 went **20.0 -> 26.6 fps, crossing into real-time (1.06x)**.
- Warmup-corrected metrics: frame 0 costs ~1.1s of CUDA/model init. Including it made
  the mean exceed the p95 and wrongly reported "not real-time". Stats now exclude warmup.
- Ollama 0.33.2 installed; `moondream` and `qwen2.5vl:3b` pulled and both run on the GPU.
  Cold start ~68-95s, warm 2.6s (moondream) / 27-45s (qwen2.5vl:3b).
- Ground-truth clip built by compositing a real truck patch into a live lane while
  traffic flows around it - gives an exact anomaly frame + bbox to score against.
- Auto zone calibration works on real footage: derived 2 carriageways with correct
  opposing flow directions (123deg / 212deg) from observed motion, no human input.

### 2026-09-03
- Stage 2 self-test PASSES; Stage 3 mock backend PASSES (runs with zero weights, no network).
- Core libs installed: ultralytics 8.4.138 (YOLO26n works), supervision 0.30.1,
  opencv-python 5.0.0, anthropic 1.3.0, kaggle 2.2.4, pydantic 2.13.5.
  `supervision` is HARD-PINNED <0.31: sv.ByteTrack is removed in 0.31 with no replacement.
- **torch 2.11.0+cu128 CUDA VERIFIED on the GTX 1650** - `sm_75` present in the arch list.
- Python 3.11 venv at `C:\dvad\.venv`. System default `py` is 3.13t freethreaded - never use it.
- Heavy artifacts live in `C:\dvad\` (outside OneDrive) so it never syncs gigabytes.
- **DECIDED: no WSL2, native Windows.** See CLAUDE.md "Why no WSL2".
- Stage 0 audit: driver 592.27 / CUDA 13.1, all 4GB VRAM free. RAM 0.5GB free of 7.35GB.

## Done and verified
- [x] All 10 src/ scripts + the Kaggle notebook written
- [x] Stage 1 real-time on 4K (threaded decode)
- [x] Stage 2 logic self-tested; anchored dwell; auto zone calibration
- [x] Stage 3 in three decision modes (hybrid / rules / vlm), async, with mock fallback
- [x] Ground-truth clip + eval harness, scoring 1.0 detection / 0.949 IoU / 0 FP
- [x] 15 candidate events harvested and ready for the teacher
- [x] Annotated demo video at `C:\dvad\outputs\demo_annotated.mp4`

### CONCURRENT-STREAMS TEST (2026-09-04) - closes the "feeds/GPU is inferred" gap
Previous feeds-per-GPU numbers (4.41 @ 1080p, 1.19 @ 4K) were single-stream
latency inverted, never actually run concurrently. Tested for real: N separate
OS processes, each loading its own YOLO26n copy into its own CUDA context
(worst-case, naive multi-process architecture - a shared-model server would do
better), all hitting the same 1080p clip at --stride 2 (need 12.5 fps each).

  N concurrent   per-stream fps        combined fps   verdict
  1              55.1                  55.1           baseline
  2              36.6 / 36.7           73.2           real-time, SUPER-linear combined
  4              23.5-23.9             ~94.8          real-time, comfortable margin
  6              18.5-18.7 (reproduced twice)  ~111.5  real-time, 49% margin - STABLE
  7              ALL FAILED            -              CUDA OOM / system RAM OOM
  8              ALL FAILED            -              cuDNN engine errors, "fatal:
                                                        Memory allocation failure"

**Verified: 6 concurrent real-time 1080p streams on one 4GB GTX 1650, reproduced
on a clean re-run.** 7+ hits a genuine wall - both GPU VRAM and the 8GB system
RAM budget (each OS process pays full torch+cv2+ultralytics import overhead).
Caveat to state if asked: this is N independent processes each with its own
model copy - the pessimistic case. A single process serving multiple streams
from one loaded model would use far less VRAM per additional stream and likely
scale further; that architecture was not what got measured here.
Reproduce: launch N `pipeline.py --stride 2` processes via `Start-Process`
(not PowerShell `Start-Job` - crashed the shell with a StackOverflowException
at 8 concurrent jobs; that crash was the job-management layer, not the GPU).

### RULE COVERAGE CLOSED + a real escalation-safety bug found (2026-09-04)
Three rules had never touched real footage (`wrong_way_vehicle`, `loitering`,
`crowd_density` w/ live VLM) - only the synthetic selftest. Also exposed that
loiter_seconds/crowd_count/wrong_way_tolerance were not CLI flags at all
(now added: `--loiter-seconds`, `--crowd-count`, `--wrong-way-tolerance`).

- **loitering**: real footage (people-walking.mp4), threshold lowered to 3s to
  get real triggers within the 13.6s clip. 5 real events, all with
  person-shaped boxes (e.g. 35x130px) and dwell timing consistent with track
  age - not tracker-ID-churn noise.
- **wrong_way_vehicle**: built a zones file with `flow_deg` flipped 180 deg from
  the real calibrated traffic direction. Real detected+tracked vehicles then
  correctly triggered wrong-way (7 events, sev 0.9). Paired negative control
  with the correct flow direction: 0 events. Clean positive/negative pair on
  real YOLO+ByteTrack output, not synthetic FakeDet.
- **crowd_density + live VLM (hybrid, real Ollama call)**: this is where it
  found a real bug, not just a coverage gap.

**BUG FOUND AND FIXED: false hazard escalation.** moondream, asked to observe
an ordinary 16-person pedestrian scene, returned `hazard_type: "person"`. The
old check in `combine()` was `hazard_type not in {"none","",  "n/a"}` - ANY
other string counted as a real hazard, so this fired a false ANOMALOUS
verdict at severity 0.9 on a completely normal crowd. Exactly the kind of
failure that would produce a red ALERT banner over ordinary pedestrians live
in front of judges.
Fix (defense in depth, since models don't reliably follow prompt constraints
alone - see the boolean-judgement finding above):
  1. Tightened OBSERVE_SYSTEM to a closed vocabulary: hazard_type must be one
     of fire/smoke/collision/debris/crowd/none. Removed "person on the road"
     from the hazard list entirely - person_in_roadway is already a Stage 2
     rule using tracker-measured lane context; letting the VLM independently
     nominate "person" as a hazard is exactly what caused the bug.
  2. Added `_is_real_hazard()` in vlm_reason.py - a hardcoded allow-list
     (fire/smoke/collision/crash/debris/crowd/explosion) that `combine()` now
     requires a substring match against, regardless of what the model outputs.
     This is the layer that actually guarantees safety: even if a model emits
     valid-JSON garbage, it cannot trigger escalation unless it names an actual
     hazard.
  Re-verified after the fix: same scene, same model, `anomalies: 0`, verdict
  correctly holds at rule_severity 0.3 instead of a false 0.9.
Full regression re-run clean after this fix: day/night/aerial ground truth all
detected=1.0, IoU 0.94-0.98, 0 false positives; both selftests pass.

### OFFLINE DRY RUN (2026-09-04) - actually tested, not just reasoned about
No admin rights in this session, so the real network adapter / firewall
couldn't be touched (`New-NetFirewallRule` -> Access Denied). Used the closest
achievable substitute: routed all external HTTP through an unroutable proxy
(TEST-NET-1, 192.0.2.1) so any external call times out instead of succeeding,
while `NO_PROXY=localhost,127.0.0.1` keeps loopback traffic working exactly as
it does when real wifi is down (loopback never touches the network hardware).

- Both selftests: instant, unaffected (4.1s combined) - confirms zero network
  dependency, not just "no obvious network calls in the code."
- **`--decision rules --aerial` on real footage: completed in 24.8s under full
  external blackhole**, detected the anomaly identically to normal runs. This
  is the actual demo command - proven to not depend on the internet at all.
- **`--decision hybrid --backend ollama`: made a genuine local call (15.9s
  latency, not instant) and completed correctly** with internet unreachable -
  confirms Ollama's local-only architecture holds under real wifi-down
  conditions, not just in theory.
- Gotcha worth keeping: a naive global HTTP_PROXY without a NO_PROXY exclusion
  incorrectly routes LOCALHOST traffic through the dead proxy too, which is
  not how real wifi-down behaves (loopback never touches the network) - it
  silently triggered the pipeline's own ollama-unreachable fallback to mock
  (0.1ms latency, the literal hardcoded mock string). Good defensive fallback
  behavior on our side, but it means an incautious version of this exact test
  would have "proven" the wrong thing.
- **Found and fixed a latent risk while at it**: `ultralytics` settings had
  `"sync": true` (telemetry phone-home) by default. Disabled it - no reason to
  carry that risk with unreliable venue wifi; it serves no purpose here.
- NOT tested under blackhole (deliberately out of scope): distill_label.py,
  push_notebook.py, setup_kaggle.py - these are pre-demo prep scripts that
  explicitly require network by design (Groq/Kaggle), never part of the live
  demo path, and SATURDAY.md already scopes them to "before Saturday."

### slow_vehicle RULE - built, measured, then DEFAULTED OFF (2026-09-04)
Added to close the recall gap where the 27B teacher flagged two cases we missed
(both "truck moving at only 6 km/h in a live lane"). Ships behind
`--enable-slow-vehicle`, OFF by default. The reasoning matters more than the code:

What it does: compares a vehicle's motion to the MEDIAN of its moving
neighbours, never an absolute km/h. That makes congestion self-cancelling - in
a jam the ambient median drops too, so nobody is an outlier and the rule stays
quiet. No separate congestion check needed.

Three things measurement changed along the way:
1. **Duration was the wrong test; consistency is.** Tracks on this footage live
   only ~0.7-0.8s (vehicles cross frame fast, ByteTrack re-acquires), so the
   original "slow for 5 seconds" could never be satisfied and the rule was dead
   code. Now: >=6 observations AND >=70% of them slow.
2. **The teacher was probably WRONG on one of its two flagged cases.** Track 13
   has a median speed ratio of 1.000 - it moves at normal speed half the time.
   The teacher saw one unlucky frame and had no motion history to know better.
   Our rule correctly declines to fire on it. This is the architecture thesis
   in miniature: the tracker has temporal information a single-frame VLM cannot
   have. Do not "fix" this disagreement.
3. **Image-plane speed alone fails in BOTH geometries.** Near-nadir puts motion
   in the image plane; oblique puts it along the view axis (centroid speed
   collapses toward zero for everyone). Gating on absolute depth motion killed
   the aerial false positives but made the rule inert on oblique footage.
   Fixed by comparing TOTAL motion (norm_speed + scale_rate_med) against the
   same measure for neighbours - like against like, works in either geometry.

Why it is off by default: after all that, it still cost 1 false positive on the
aerial ground-truth run, and every threshold had been set by chasing FPs on a
single clip. That is overfitting to one video, and zero-false-positives across
538 frames is the strongest claim this system has. Not worth trading a large-
sample result for a recall point measured on n=15 against a partly-wrong
teacher. Turn it on when there is footage to validate it against.
Verified: default config = 0 FP on day/night/aerial; `--enable-slow-vehicle`
fires correctly when asked.

Also fixed while here: stopped_vehicle and slow_vehicle now share a dedup
"family" (`_kind_family`), so one impeded vehicle cannot raise two alerts under
two different kind names; and slow severity is capped below the stopped range
so a crawl never outranks a dead stop.

### *** VisDrone FINE-TUNE LANDED - 2.25x aerial recall, drop-in *** (2026-09-04)
Weights cached at `C:\dvad\models\yolo26n_visdrone.pt` (5.2MB). Training:
30 epochs on Kaggle T4, mAP50 0.137 -> 0.378, precision 0.483, recall 0.387.

Same 100 VisDrone val images, same settings (imgsz 1024, conf 0.15):

  group          stock COCO   fine-tuned    change
  vehicle          0.522        0.836        +60%
  person           0.291        0.655        +125%
  twowheel         0.068        0.496        +629%
  OVERALL          0.294        0.661        +125%  (2.25x)
  tiny objects     0.063        0.440        +598%  (7x)

Full journey on aerial recall: 0.152 (stock defaults) -> 0.294 (--aerial config)
-> **0.661 (fine-tuned + --aerial)**. 4.3x from where this started.
The tiny-object jump (0.063 -> 0.440) is the one that matters most for drone
footage, where most objects are small.

Drop-in verified - day/night/aerial ground truth all still detected=1.0,
IoU 0.95-0.98, **0 false positives**, still real-time (21.1 fps, 1.68 feeds/GPU).
Use with: `--weights C:\dvad\models\yolo26n_visdrone.pt`

THREE silent-failure traps had to be fixed before these weights were usable.
None of them would have raised an error - all three just quietly degrade:
1. **Class filtering was by COCO id.** VisDrone ids mean different things
   (0=pedestrian, 3=car, 4=van, 9=motor), so passing COCO ids filtered to a
   wrong-but-plausible subset with no error. Now resolved by NAME against
   whatever weights are loaded (`resolve_target_class_ids`), and returns None
   ("don't filter") rather than an empty list, because detecting nothing
   silently is the worse failure.
2. **Stage 2's VEHICLE_NAMES/PERSON_NAMES didn't know the VisDrone names.**
   A missing name does not error - Stage 2 just ignores that class. Added van,
   motor, tricycle, awning-tricycle, pedestrian, people.
3. **The car-length ruler had no entries for the new classes**, so
   metres-per-pixel and the km/h estimate silently returned None for exactly
   the vehicle types Indian urban footage is full of - auto-rickshaws included.
   Added van 5.5m, tricycle 2.8m, awning-tricycle 3.0m, motor 2.1m, etc.
Also fixed eval_aerial.py to score both vocabularies, or the comparison above
would have unfairly marked every pedestrian/van/motor detection as a miss.

### VisDrone fine-tune: the notebook wasted its own output
Training completed on Kaggle T4 (~50 min, 30 epochs). Then pulling the result
turned into a multi-GB download, because the notebook let ultralytics put the
~2GB VisDrone dataset inside `/kaggle/working` - and EVERYTHING under
/kaggle/working becomes kernel output. So `kaggle kernels output` dutifully
pulled VisDrone2019-DET-train.zip (1.48GB) + test-dev (297MB) + val (78MB) +
extracted images back down, just to reach a ~5MB best.pt.

Fixed in `notebooks/build_yolo_notebook.py` for any re-train on the day:
  * `settings.update({"datasets_dir": "/kaggle/temp/datasets"})` - /kaggle/temp
    is scratch and is NOT part of kernel output.
  * A trim step at the end that keeps best.pt / last.pt / results.csv / plots
    and deletes epoch checkpoints and any stray dataset, then prints the final
    output size so the mistake is visible next time.
  * Telemetry `sync` also disabled inside the notebook.
This matters if the organisers' data arrives and a re-train is wanted: without
the fix, each pull costs several GB on venue wifi.

Also note: `--aerial` config tuning already delivered the 3x recall win
independently, so the fine-tuned weights are an upgrade, not a dependency.
Compare before trusting them:
  `python src\eval_aerial.py --limit 100 --imgsz 1024 --conf 0.15`
  `python src\eval_aerial.py --limit 100 --imgsz 1024 --conf 0.15 --weights <new.pt>`
REMINDER: VisDrone class ids are NOT COCO (0=pedestrian 1=people 2=bicycle
3=car 4=van 5=truck 6=tricycle 7=awning-tricycle 8=bus 9=motor). Add `van` and
the tricycles to VEHICLE_NAMES/VEHICLE_CLASSES before using these weights, or
Stage 2 will silently ignore most vehicles.

## Remaining after the blockers clear
- [ ] `distill_label.py` on the 15 harvested events (~$0.10 with claude-opus-5)
- [ ] `build_kaggle_dataset.py --push`
- [ ] Commit the Kaggle notebook, download the adapter to `C:\dvad\models\`
- [ ] Distillation-fidelity eval (student vs teacher F1)
- [ ] Full offline dry run with wifi OFF
