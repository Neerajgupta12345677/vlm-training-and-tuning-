# Handover — switching to Cursor

Written 2026-09-04, ~15:00 IST. Hackathon is 2026-09-05, 09:00, FlytBase Labs
Pune (or online) — **~18 hours out**. This file is the fast-start entry point;
it does not repeat what's already documented elsewhere, it tells you what
order to read things in and what's true right now.

## Read order (5 min, don't skip)

1. **This file** — orientation + the one blocking action
2. `CLAUDE.md` — hard constraints (paths, no-WSL2 decision, hardware, model
   choices) and the architecture. Read the whole thing once; it says "do not
   redesign without asking" for a reason — those decisions were made under
   measurement, not preference.
3. `PROGRESS.md` — the actual build log, **newest entries at the top**. This
   is where every "why" lives: every bug found, every number measured, every
   design reversal and the reasoning behind it. If something looks like it
   could be done differently, check here first — it's very likely already
   been tried and rejected for a documented reason.
4. `SATURDAY.md` — the runbook for tomorrow. Copy-pasteable commands.
5. `DEMO.md` — the one-page pitch card with every number that's safe to quote
   to judges, and which numbers are NOT safe to quote (see "Honest gaps" below).

## The one blocking action

The organisers released the real dataset today (doc:
`AHC Visual Intelligence Hackathon (1/2/3).pdf` in `Downloads/`, also copied
into the repo root). It has:
- 12 exact class strings (`normal`, `traffic_accident`, `traffic_congestion`,
  `stalled_or_broken_down_vehicle`, `vehicle_blocking_traffic`,
  `wrong_way_driving`, `road_spill_or_debris`, `waterlogging_or_flood`,
  `fire`, `smoke`, `fighting_or_violence`, `loitering_or_suspicious_presence`)
- A documented CSV schema for `ground_truth.csv` / submissions
- A **public test set with its own `ground_truth.csv`** — meaning we can score
  ourselves on real data before the private leaderboard even opens
- `description_summary` per training video — real distillation targets, far
  better than our synthetic n=15 Groq-labelled set

**Status as of this file: the 15–17GB train+test pack has NOT landed in
`C:\dvad\data\ahc\` yet** (checked immediately before writing this — the
directory exists and is empty). The user is downloading it via a mirror link
in the dataset doc (Google Doc, auth-gated, can't be fetched by an agent).

**The moment it lands**, extract to `C:\dvad\data\ahc\` (NOT the OneDrive repo
folder — that would trigger OneDrive sync on 15GB and risks saturating venue
wifi) so you get:
```
C:\dvad\data\ahc\train\<class_name>\videos\*.mp4 (+ videos.csv, ground_truth.csv)
C:\dvad\data\ahc\test\videos\*.mp4 (+ videos.csv, ground_truth.csv)
```
Then run, in this order:
```
:: 1. score ourselves on the public test set — this is the single most
::    valuable thing to do first, it's a REAL number, not self-graded synthetic
python src\run_ahc_dataset.py --data_dir C:\dvad\data\ahc --split test --out C:\dvad\outputs\predictions.csv
python src\score_submission.py --gt C:\dvad\data\ahc\test\ground_truth.csv --pred C:\dvad\outputs\predictions.csv

:: 2. pull real distillation labels from train/ (replaces the synthetic Groq set)
python src\run_ahc_dataset.py --data_dir C:\dvad\data\ahc --extract-labels-only --out C:\dvad\data\ahc_distill_labels.jsonl
```
`run_ahc_dataset.py` and `score_submission.py` were built and self-tested
tonight against a synthetic tree matching the documented schema exactly
(18+14+9 assertions passing) — they have never touched real data yet. First
real run may surface something the synthetic fixture couldn't (see "Open
issues" below for one already-known example of this exact pattern).

**Paused, deprioritised, resumable if wanted:** a Kaggle fire/smoke dataset
(D-Fire, 3GB) is half-downloaded at
`C:\dvad\data\datasets\dfire\smoke-fire-detection-yolo.zip.kaggle-partial`
(1.05GB of 3.05GB). It was paused because the organisers' real dataset
strictly beats it (same distribution as the actual test set). Only resume if
the real fire/smoke coverage in `train/` turns out to be thin.

## Critical invariants — regression-check these before any commit

Zero false positives across day/night/aerial on our ground-truth clip is the
single strongest, most load-bearing claim this project has. Run this after
ANY change to `context_state.py`, `detect_track.py`, `ego_motion.py`, or
`vlm_reason.py`:
```
python src\context_state.py --selftest
python src\vlm_reason.py --selftest --backend mock
python src\pipeline.py --source C:\dvad\data\vehicles_stopped.mp4 --zones C:\dvad\data\vehicles_stopped_zones.json --decision rules --aerial --stop-seconds 5 --cooldown 8 --stride 2 --out C:\dvad\outputs\reg.jsonl
python src\eval.py --ground-truth C:\dvad\data\vehicles_stopped_ground_truth.json --predictions C:\dvad\outputs\reg.jsonl
```
Expected: both selftests `PASS`, `detected: 1 (rate 1.0)`, `best IoU: ~0.98`,
`FALSE POSITIVES: 0`. Also spot-check every `src\*.py --help` still exits 0
(19 scripts as of this commit) — a broken import anywhere breaks the whole
chain silently otherwise.

## Environment quick reference

- **Never use bare `python`/`py`** — system default is 3.13 freethreaded, has
  no ML wheels. Always `C:\dvad\.venv\Scripts\python.exe`.
- Code/docs live in this OneDrive repo folder (small, git-tracked). Everything
  heavy — venv, models, videos, datasets — lives in `C:\dvad\` outside
  OneDrive. Never let anything large land inside the repo folder.
- Credentials: Kaggle uses a standalone `KGAT_...` token at
  `~/.kaggle/access_token` (not the legacy `kaggle.json` — verify with
  `python src\setup_kaggle.py --verify-only`). Groq key is a **user-scope**
  env var (`GROQ_API_KEY`) — a fresh terminal picks it up automatically; if a
  script can't see it, that's the first thing to check, not a code bug.
- Ollama is running locally with `qwen2.5vl:3b` and `moondream:latest`
  already pulled — confirm with `curl http://localhost:11434/api/tags`.
- Git: `main` branch, pushed to
  `github.com/Neerajgupta12345677/vlm-training-and-tuning-`, clean as of
  commit `ebb9c04`. Commit convention used throughout: explain WHY in the
  body, cite the measurement that justified the change, not just WHAT
  changed.

## File map (`src/`)

| File | What it does |
|---|---|
| `common.py` | Shared paths, video I/O, threaded frame reader |
| `detect_track.py` | Stage 1: YOLO + ByteTrack detection/tracking |
| `ego_motion.py` | Camera-motion compensation (LK flow + RANSAC affine) — **critical**, see below |
| `context_state.py` | Stage 2: all the dwell/zone/congestion/crowd rules — the core logic |
| `vlm_reason.py` | Stage 3: VLM prompts, hazard vocabulary, escalation logic |
| `pipeline.py` | Wires all three stages together, the main entry point |
| `demo.py` | One-command demo driver (calibrates zones, runs, reports) |
| `eval.py` | Scores a run against a ground-truth clip (IoU, false positives) |
| `eval_aerial.py` | Scores raw detector recall against VisDrone |
| `calibrate_zones.py` | Auto/manual/whole-frame zone calibration |
| `get_sample_data.py` | Builds synthetic test clips (stopped vehicle, hazards, camera motion) |
| `label_map.py` | Our event kinds → the 12 official AHC class strings |
| `submission.py` | events.jsonl → AHC submission CSV rows |
| `score_submission.py` | Scores predictions.csv against ground_truth.csv |
| `run_ahc_dataset.py` | Batch-runs the pipeline over the AHC train/test tree |
| `distill_label.py` | Teacher-labels frames via Groq/OpenRouter/Anthropic |
| `build_kaggle_dataset.py` | Packages labelled frames for Kaggle upload |
| `push_notebook.py` | Pushes/runs/pulls Kaggle training notebooks from the CLI |
| `setup_kaggle.py` | Installs/verifies the Kaggle token |

## Open issues, ranked

1. **One finding not chased tonight** — `stopped_vehicle` now fires once
   instead of repeatedly on the canonical clip, coinciding with adding
   ego-motion compensation. Detection accuracy is unaffected (still 1.0 / 0.981
   IoU / 0 FP) — only `end_time_sec` on long-running anomalies may be
   understated. Needs real footage to diagnose properly; see the bottom of
   `PROGRESS.md`'s "SUBMISSION PIPELINE BUILT" entry for the full trace.
2. **`is_anomaly` CSV bool format is unverified** — we write `True`/`False`;
   the real `ground_truth.csv` might use `true`/`false` or `1`/`0`. Our local
   scorer parses all three tolerantly, so *our* scoring won't break — but if
   there's a real submission portal, check its expected format before
   uploading.
3. **`slow_vehicle` and `--watch-for` open-vocabulary querying are real but
   under-exercised** — both work (measured), both are off by default /
   opt-in. Worth turning on once real footage shows they're needed, with a
   fresh false-positive check each time (see `PROGRESS.md` on why
   `slow_vehicle` defaults off).
4. **Appearance hazards (fire/smoke/flood/debris) are structurally
   unvalidated** — our synthetic composites were shown to be inadequate
   proxies (the VLM correctly called a fake flood overlay "in good
   condition"). Do not quote a detection rate for these until real footage
   exists. The `train/fire/`, `train/smoke/`, `train/waterlogging_or_flood/`
   folders (once downloaded) are the real fix.
5. **VisDrone fine-tuned weights** exist at `C:\dvad\models\yolo26n_visdrone.pt`
   (2.25x aerial recall, drop-in verified) but use VisDrone's class
   vocabulary (van/motor/tricycle/etc, not COCO) — already wired through
   `detect_track.py`'s name-based class resolution, should just work with
   `--weights` on real footage, but hasn't been tested on non-VisDrone,
   non-our-own-clips data yet.

## What "elite" still needs (from an earlier adversarial review, still true)

- No score on the organisers' actual leaderboard yet — everything above is
  self-graded, even the "real" scoring is against public data, not the
  private eval set.
- The LoRA distillation adapter (`C:\dvad\models\lora_adapter\`) is trained
  but never loaded at inference anywhere — it's a proof the training loop
  works, not a live capability. If time allows, either wire it in or be ready
  to say precisely that honestly.
- Accidents (~1s duration) are still structurally hard for a dwell-based
  system — the temporal-montage idea (tile 4 frames spanning a few seconds
  into one VLM call) was proposed but never built. Worth it if `train/
  traffic_accident/` turns out to have clips we're clearly missing.
