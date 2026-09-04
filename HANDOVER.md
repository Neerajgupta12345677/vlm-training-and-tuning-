# Handover — 2026-09-04, ~20:35 IST

Hackathon 2026-09-05, 09:00, FlytBase Labs Pune. This file is a full rewrite,
not an update — a lot happened since the last version (cascade built and
debugged, Kaggle fine-tune run and diverged). Read this file top to bottom;
it supersedes everything it says was true earlier tonight.

## The one-paragraph version

The organisers' real dataset landed and got fully merged (1845 train videos,
34 test videos, all 5 download mirrors accounted for, verified file-by-file).
A MobileNet classifier trained on it locally and, once properly tuned, is
the **safe submission at macro-F1 0.188** — file:
`C:\dvad\outputs\predictions_final.csv`, already verified fresh. A more
ambitious cascade (classifier + VLM adjudication on contested clips) was
built, hit two real serving bugs, both got fixed, and it now runs reliably —
but nets out at 0.181-0.183, not yet beating the classifier alone. A Kaggle
QLoRA fine-tune of Qwen2.5-VL-3B was also attempted and **diverged**
(`loss: nan`) — that adapter is unusable and is not wired into anything.

## Read order

1. **This file.**
2. `CLAUDE.md` — hardware/path constraints, the original 3-stage architecture
   (still real, still the demo story, no longer what's scored).
3. `PROGRESS.md` — newest-first build log, every "why" lives there.
4. `SATURDAY.md` — runbook, written before tonight's cascade work; still
   correct on the classifier-only path (§1-4), stale on anything VLM-related.
5. `DEMO.md` — pitch card, **numbers stale**, see "safe to quote" below before
   using it.

## 1. Is the data fully used?

**Yes for the classifier, no (deliberately) for the fine-tune.**

- **MobileNet classifier**: trained on **all** 1845 unique train videos.
  Verified tonight, file-by-file, across all 5 downloaded "mirror" zips (which
  turned out to be overlapping PARTIAL exports of the same Google Drive
  folder, not 5 identical copies — each mirror alone is missing classes; the
  union of all 5, deduplicated by filename, is what's merged into
  `C:\dvad\data\ahc\train\`). Zero videos lost in the merge — checked per
  class.
- **VLM fine-tune dataset** (`vlm_ft/`): a deliberate SUBSET — 3 of the 8
  cached frames per video, `normal` capped at 400 videos so it can't drown out
  rarer classes. 4,371 samples. This is moot now since the run diverged, but
  if retried, `--frames-per-video 8 --max-per-class 600` uses more of what's
  already cached, no re-decoding needed.
- **Test set**: 33/34 videos present (T030 is missing from the organisers'
  own pack, not on our end — every submission-writer already backfills a
  `normal` row for it).

## 2. Running everything locally, without Claude, and the GitHub question

**Every script here is a standalone CLI** (`python src\X.py --help` works on
all 27 of them) — nothing requires an AI in the loop to execute. Tonight I
was just the one typing commands and reading output; the commands themselves
are ordinary.

**GitHub is not, and does not need to be, in the Kaggle path.**
`src\push_notebook.py` pushes directly from your laptop to Kaggle via the
`kaggle` CLI (`kaggle kernels push`) — there is no GitHub hop, and adding one
would be an extra step for no benefit (Kaggle notebooks don't pull from
GitHub in this workflow; the notebook JSON itself is uploaded directly).
GitHub is used here only for what it's normally for: versioning the `src/`
code and these docs (see `git log`).

**The exact command sequence to reproduce the safe submission, start to
finish, with zero Claude involvement:**
```
set PY=C:\dvad\.venv\Scripts\python.exe

:: 1. train the classifier (skips already-cached frames on a rerun, ~15-20 min from scratch)
%PY% src\train_appearance.py --data_dir C:\dvad\data\ahc --epochs 12

:: 2. score every test video (~20s for 33 videos)
%PY% src\appearance_classifier.py --data_dir C:\dvad\data\ahc --split test ^
  --weights C:\dvad\models\appearance11.pt --dump C:\dvad\outputs\app_scores.json

:: 3. pick the best decision rule against the REAL scorer (arithmetic, no GPU, seconds)
%PY% src\tune_appearance.py --scores C:\dvad\outputs\app_scores.json ^
  --gt C:\dvad\data\ahc\test\ground_truth.csv --write C:\dvad\outputs\predictions.csv

:: 4. verify the score
%PY% src\score_submission.py --gt C:\dvad\data\ahc\test\ground_truth.csv --pred C:\dvad\outputs\predictions.csv
```
Expect macro-F1 **~0.188** at step 4 (may vary slightly run-to-run: training
has randomness in frame sampling/augmentation/shuffling; re-running step 1
is not guaranteed to reproduce the exact same checkpoint — see gap #1 below).

**If you want the Kaggle fine-tune path (currently broken, see §Kaggle
below) run purely from a terminal, no Claude:**
```
%PY% src\build_vlm_dataset.py --data_dir C:\dvad\data\ahc
kaggle datasets create -p C:\dvad\data\vlm_ft -r zip --dir-mode zip
%PY% src\push_notebook.py --push --slug <your-slug> --title "<your title>" ^
  --dataset dvad-ahc-vlm-ft --accelerator NvidiaTeslaT4
%PY% src\push_notebook.py --status --slug <your-slug>      :: poll this
%PY% src\push_notebook.py --pull  --slug <your-slug>        :: once COMPLETE
```
Fix the learning rate first — see gap #2 below — before spending another
~75 minutes of GPU-hours on a repeat of tonight's divergence.

## 3. The flow, in plain words

**Training (offline, done once, before the demo):**
Take the 1845 labelled training videos. Pull 8 representative snapshots out
of each one. Show a small image-recognition model (MobileNet — a few million
parameters, not a language model) thousands of these snapshots along with
their correct label ("this is fire", "this is normal", ...), and let it
adjust itself until it gets good at guessing correctly on snapshots it
wasn't shown during training. That adjusted model is saved to disk
(`appearance11.pt`) and is what actually runs at test time.

Separately, an attempt was made to teach a much bigger model (a
vision-language model, Qwen2.5-VL) to imitate real human-written
descriptions of these same videos, using a rented GPU on Kaggle. That
training run broke partway through (the mathematical signal it learns from
became invalid — "NaN") and produced a model that doesn't reliably work, so
it isn't being used.

**Inference (what happens to a NEW test video):**
Grab 8 snapshots from the video, show all 8 to the trained MobileNet, and
average its opinion across them. If it's clearly sure of one answer, use
that answer. If it's genuinely torn between two or three answers, show ONE
snapshot to the (untrained, off-the-shelf) big vision-language model and ask
it to pick from the MobileNet's short list — but this extra step currently
isn't reliably making things better than just trusting MobileNet's own best
guess, so tonight's actual submitted answer comes from MobileNet alone.

## 4. Measured results (real public test set, 33/34 videos scored)

| run | macro-F1 | exact-set acc | is_anomaly acc | notes |
|---|---|---|---|---|
| rules-only (motion tracker, no classifier) | 0.023 | 0.118 | 0.077 | can't see appearance-only classes at all |
| classifier, epoch-1 ckpt (**lost**, cannot reproduce) | 0.256 | 0.353 | 0.404 | historical only — checkpoint overwritten |
| classifier, epoch-9 ckpt (**current**, tuned global rule) | **0.188** | 0.382 | 0.385 | **the safe submission, verified fresh** |
| classifier + per-class thresholds, fit on test (in-sample) | 0.289 | 0.441 | 0.442 | looks great, is fake — see below |
| classifier + per-class thresholds, leave-one-video-out | 0.230 | — | — | the honest number for the row above |
| classifier + per-class thresholds, fit on held-out VAL | 0.126 | — | — | a second, independent way of trying this, also failed |
| cascade: classifier + base VLM, gate as-run | 0.181 | 0.382 | 0.365 | both real serving bugs fixed; net ≈ tied with classifier alone |
| cascade: best replayable gate (fewer VLM calls) | 0.183 | 0.382 | 0.385 | still short of 0.188 — the gate isn't the bottleneck |
| cascade + fine-tuned adapter | **not attempted** | | | adapter diverged, never wired in |

**Why the epoch-1 number is gone and matters:** `train_appearance.py` keeps
only the single best-by-val-recall checkpoint, overwriting on every
improvement. Val recall climbed 0.650→0.742 across training, but test
macro-F1 went the OPPOSITE direction (0.256→0.188) — a real, measured domain
shift between the training-pool validation split and the reserved test
source, not noise. **This means "more training = better" is false on this
dataset**, and there is currently no way to get back the better-scoring
early checkpoint. See gap #1.

**Per-class detail on the current safe submission (0.188 run):**

| class | F1 | GT support | status |
|---|---|---|---|
| loitering_or_suspicious_presence | 0.8 | 4 | working |
| traffic_congestion | 0.667 | 4 | working |
| traffic_accident | 0.267-0.364 | 7 | working, imprecise |
| normal | 0.286-0.5 | 6 | working |
| fighting_or_violence | 0.0 | 3 | **dead** |
| fire | 0.0 | 2 | **dead** |
| road_spill_or_debris | 0.0 | 3 | **dead** |
| smoke | 0.0 | 2 | **dead** |
| stalled_or_broken_down_vehicle | 0.0 | 1 | **dead** (deliberately not modelled, see gap) |
| vehicle_blocking_traffic | 0.0 | 2 | **dead** |
| waterlogging_or_flood | 0.0 | 2 | **dead** |
| wrong_way_driving | 0.0 | 1 | **dead** |

**The dead classes are not a modelling failure** — the classifier's own
held-out validation F1 on these exact classes is high (waterlogging 0.914,
vehicle_blocking 0.772, fighting 0.680 — see `val_thresholds.json`). They die
at test time because plain top-1 argmax requires beating 11 competitors, and
domain shift plus class imbalance means they usually don't. This is gap #3.

## 5. Full gap/issue list, ranked by leverage

**#1 — No checkpoint history during training.** `train_appearance.py`
overwrites `appearance11.pt` on every val-recall improvement. Given the
measured val-vs-test divergence above, this means the single artifact kept
is optimised for the WRONG signal. **Fix**: save every epoch (or every
improving epoch) to a distinct filename, and after training, score EACH
candidate against the real test set (cheap — `appearance_classifier.py
--dump` is ~20s) before picking one. This is the single highest-leverage
unfixed issue — it may recover some or all of the gap back to 0.256.

**#2 — Kaggle fine-tune diverged (`loss: nan`).** Root cause not confirmed
(per-step loss wasn't captured by Kaggle's log streaming — only the final
NaN is visible). Most likely cause: `lr=2e-4` in `notebooks\build_notebook.py`
`CONFIG` is aggressive for 4-bit QLoRA on a vision-language model; try
`5e-5` or `1e-4` first. **Before any retry**, patch the notebook to also
write loss to `/kaggle/working/loss.jsonl` every step (not just rely on
Kaggle's captured stdout) so a second divergence is caught early rather than
discovered only after ~75 minutes.

**#3 — The dead-class argmax problem is unsolved.** Three fixes tried,
three failed for three different reasons (test-set overfit, val-set domain
shift, VLM adjudication netting a wash). Untried ideas, cheapest first:
  - Multi-label thresholding instead of single-label argmax — assert EVERY
    class whose probability clears a low, class-specific floor, rather than
    only the winner. Given ground truth allows multiple labels per video,
    this may be a more honest match to the task than forcing one winner.
  - Loosen the cascade's VLM system prompt — it currently says "be
    conservative: if nothing unusual is visible, choose normal", which
    tonight's data showed produces a strong bias toward `normal` (9 of 11
    label changes moved toward `normal`, recovering none of the 6 dead
    classes). Try removing that line or reversing its bias.
  - More training data for the thinnest classes specifically (fire: 77
    videos, weakest val F1 of the six dead classes at 0.500).

**#4 — `--label-source hybrid` in `run_ahc_dataset.py` has still never been
run end to end.** It's meant to recover `stalled_or_broken_down_vehicle`
(deliberately unmodelled — only 4 training videos) via the motion rules
while the classifier owns everything else. Costs ~20-40s/video for the
motion pipeline. Untested whether it nets positive.

**#5 — The cascade's VLM leg is a wash, and the fresh-load fix costs real
latency.** `--fresh-load` (forces Ollama to reload the model every call) is
what fixed the reliability bug, but it adds a model-load's worth of latency
per call, and the *underlying reason a fresh load was needed at all*
(session-state accumulation across a long-running Ollama server) was never
root-caused — only worked around. Worth understanding if the cascade is
pursued further, especially since the eventual fine-tuned adapter shares
this exact serving path.

**#6 — Montage/temporal evidence to the VLM is confirmed non-viable on this
hardware**, not just difficult. A 2x2 composited grid degenerates the model
into repeated-token garbage at every size tested (512-896px), and a true
multi-image call times out outright. This closes off the most-promising
architectural idea from tonight's design discussion (giving the VLM motion
evidence) unless the serving backend changes (e.g. HF/PEFT instead of
Ollama, untested) or better hardware becomes available.

**#7 — LoRA adapter (even a working one, hypothetically) has no path to
inference.** No merge/GGUF/Modelfile step exists in the repo. If gap #2 ever
produces a working adapter, this is unbuilt from scratch — budget real time,
it's historically the fiddliest step in this kind of pipeline (per
`CLAUDE.md`'s own note).

**#8 — `predictions_SAFE_0.188.csv` was caught mislabeled tonight** (an
earlier `cp` grabbed a stale 15:44 file that actually scores 0.023, not
0.188) **and has been fixed**, but it's a reminder: always re-verify a
"safe copy" with `score_submission.py` before trusting its filename.

**#9 — No score exists on the private leaderboard.** Everything above is
self-measured against the organisers' PUBLIC test set, which they explicitly
provide for exactly this purpose, but it is not the graded set.

**#10 — DEMO.md is now stale** on every score-related number (still quotes
the old rules-only pipeline framing). Needs a rewrite pass before presenting
to judges — see "safe to quote" below for what's currently true.

## Findings from tonight you should not re-litigate (all measured, not guessed)

- **Per-class decision thresholds have now failed THREE different ways** —
  fit on test (overfits, 0.230 held out), fit on val (0.126, doesn't
  transfer — real domain shift), and implicitly via the cascade's gate
  (0.183 best case, still short). Use the plain global rule.
- **A composited multi-frame montage reliably breaks qwen2.5vl:3b via
  Ollama** on this 4GB/no-tensor-core card — confirmed on a clean,
  uncontended GPU, independent of the session-flakiness bug (which is a
  separate, since-fixed issue). Single frames work.
- **`kill $BGPID` in Git Bash does not reliably kill a Windows child
  process.** A cancelled test run kept going unseen for ~20 minutes tonight,
  pinning the GPU and confounding an entire diagnostic chain. Use
  `nvidia-smi --query-compute-apps` + `Stop-Process -Id <pid> -Force` to
  actually confirm and clear GPU-holding processes.
- **Speed-based congestion detection cannot work on this footage** —
  `diag_speeds.py` showed a genuinely normal clip reading as MORE congested
  than both ground-truth congestion clips at every threshold tested. Box
  jitter at low resolution swamps the speed estimate. This is why
  congestion moved to the classifier.
- **The test set is three datasets in one costume** — clip lengths and frame
  rates vary wildly (5.7s to 629s, 2fps to 25fps), making some clips
  structurally hostile to any motion-based rule. The classifier's
  frame-sampling approach is less sensitive to this than the motion pipeline.
- **T030 is missing from the public pack.** Every submission-writer already
  backfills a `normal` row for it.

## Numbers safe to quote to judges (verified tonight, cite the checkpoint)

- Public test set macro-F1 **0.188** (epoch-9 checkpoint, `tune_appearance.py`
  global rule, verified `2026-09-04 20:32` — always quote a number you
  reproduced yourself, not this stale copy, if it's been a while).
- Appearance classifier: MobileNetV3-Small, 2.54M params, ~6MB on disk,
  ~0.3s/clip vs 27-45s for one qwen2.5vl:3b call.
- Stage 1 (YOLO) alone, threaded decode: 26.6 fps, 1.06x real-time on 4K.
- Full motion pipeline, `--decision rules`, stride 2: 14.9 fps vs 12.5 needed.

**NOT safe to quote:**
- Anything about the private leaderboard.
- The LoRA adapter as a live capability — it diverged and is not used.
- 0.245 or 0.256 — historical, from a checkpoint that no longer exists.
- Any per-class detection rate for the six classes at F1 0.0.
- The cascade as an improvement over the classifier — measured, it currently
  isn't one.

## Environment quick reference

- **Never use bare `python`/`py`** — always `C:\dvad\.venv\Scripts\python.exe`.
- Code/docs in this OneDrive repo; everything heavy (`venv`, models, videos,
  frame caches) lives outside OneDrive under `C:\dvad\`.
- Dataset: `C:\dvad\data\ahc\{train,test}` — 1845 train videos (12 class
  folders), 33/34 test videos.
- Frame caches: `appearance_frames11` (current, 11-class, 14,976 frames) —
  `appearance_frames` (old 7-class) is superseded, do not point new tools
  at it (this was a real bug caught and fixed tonight in
  `build_vlm_dataset.py`'s default).
- Models: `appearance11.pt` (current classifier, val macro-recall 0.742),
  `appearance11_epoch9_0.742.pt` (identical weights, kept as an insurance
  snapshot — same checkpoint, training never improved past it),
  `appearance_classifier.pt` (older 7-class, do not use — its `normal` class
  wrongly contains 4 classes since promoted), `yolo26n_visdrone.pt`
  (aerial-tuned detector, unrelated to the classifier work, still good),
  `lora_adapter\` (⚠️ contains the OLD synthetic-data adapter from this
  morning, predating tonight's diverged Kaggle run — do not use either one).
- Kaggle: authenticated as `guptaneeraj123`, phone-verified (confirmed
  working tonight — GPU smoke test `dvad-gpu-smoke` completed clean).
- Ollama: `qwen2.5vl:3b` and `moondream:latest` cached locally, served on
  `localhost:11434`.

## File map — everything new or changed tonight

| File | What it does |
|---|---|
| `src\train_appearance.py` | Trains the MobileNet classifier. 11 classes, video-level split, self-healing frame cache. **Gap #1: no per-epoch checkpoint history.** |
| `src\appearance_classifier.py` | Loads a checkpoint, classifies a video or dumps per-video probabilities (`--dump`) for threshold sweeps. |
| `src\tune_appearance.py` | Sweeps the classifier's decision rule (threshold/top_k/margin) against the real scorer. `--per-class --cv` exists specifically to catch overfitting — and did. |
| `src\calibrate_thresholds.py` | Second attempt at per-class thresholds, fit on the held-out VAL split instead of test. Also failed (0.126) — kept for the documented negative result. |
| `src\diag_speeds.py` | Measures per-frame speed distributions; proved speed-based congestion detection is impossible on this footage. |
| `src\build_vlm_dataset.py` | Builds the Unsloth fine-tuning set from real organiser captions. Defaults FIXED tonight to point at `appearance_frames11`, not the stale 7-class cache. |
| `src\cascade.py` | Classifier proposes, VLM adjudicates contested clips. Two real bugs fixed (Ollama session flakiness via `--fresh-load`, prompt-parroting parser bug). Montage/temporal evidence confirmed non-viable on this hardware — defaults to single-frame. |
| `src\tune_cascade_gate.py` | Replay-sweeps the cascade's confidence gate against already-collected VLM answers, zero new calls. Localised the shortfall to the VLM's judgments, not the gate. |
| `notebooks\build_notebook.py` | Generates `finetune_kaggle.ipynb`. Patched tonight for the new `messages`-format dataset (previously wired to a dead format that would have crashed). `max_steps` raised 60→800 for the real dataset size. **`lr=2e-4` is the prime suspect for tonight's divergence — try lower first.** |
| `src\run_ahc_dataset.py` | Batch runner, `--label-source {hybrid,appearance,rules}`. Hybrid path still never run end to end (gap #4). |
| `src\score_submission.py`, `src\submission.py`, `src\label_map.py` | Submission schema, scorer, class-name mapping — all built and verified against the REAL `ground_truth.csv` tonight (earlier they were only tested on synthetic fixtures). |

## Output files worth knowing about, in `C:\dvad\outputs\`

- `predictions_final.csv` / `predictions_SAFE_0.188.csv` (identical,
  **verified 0.188**) — the current safe submission.
- `app_scores.json` — current checkpoint's raw per-video probabilities on
  test. Re-tuning or re-scoring starts here, no need to re-run the classifier.
- `val_thresholds.json`, `val_scores.json` — the failed val-calibration
  attempt's data, including the per-class val-F1 numbers that motivated the
  cascade (kept because they're genuinely informative about what the
  classifier does and doesn't know, even though the calibration itself failed).
- `trace_cascade3.jsonl` — full cascade run trace with both bugs fixed
  (real VLM answers, real classifier probabilities per video). `tune_cascade_gate.py`
  reads this.
- `train11.log` — full training log, all 12 epochs, per-class val recall
  every epoch.
