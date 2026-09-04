# Saturday runbook — Sept 5, FlytBase Labs

**Event:** Visual Intelligence Hackathon, 9:00–19:00. Build window **11:00–18:00**,
demos 18:00–19:00. Rewritten 2026-09-04 evening after the real dataset landed and
the architecture pivoted on measured evidence — the previous version of this file
described a pipeline that no longer carries the score. If you are reading a stale
copy, check `HANDOVER.md` first, it is the fast-start entry point.

Everything below is copy-pasteable. `PY` is the venv interpreter; nothing uses
bare `python` (the system default is 3.13t freethreaded, no ML wheels).
```
set PY=C:\dvad\.venv\Scripts\python.exe
```

---

## 0. The one thing to understand before touching anything

**The task as scored is multi-label CLIP CLASSIFICATION, not temporal
localisation.** Verified three ways on the organisers' own public test set:
every one of the 52 `ground_truth.csv` rows has empty `start_time_sec`/
`end_time_sec`; `train/` is one folder per class; 52 GT rows over 34 videos
means a video can carry several labels. **Do not spend build-window time tuning
dwell thresholds for precise timing** — it is not what gets scored. The
three-stage streaming cascade (YOLO+track → context rules → VLM) is still real
and still the demo story for *why* a small model can do this cheaply, but the
thing that actually produces the submitted label is a classifier, described
below.

## 1. Sanity check (run this FIRST, before anything else)

```
%PY% -c "import torch; print('cuda', torch.cuda.is_available())"
%PY% src\context_state.py --selftest
%PY% src\vlm_reason.py --selftest --backend mock
```
All three must print `PASS`/`True`. The last needs no weights and no network —
if that works, the demo survives even with the wifi dead.

## 2. If a NEW dataset drops on the day

The organisers' real pack (train+test, ~15–17GB) already has this exact shape
and it is what every script below expects — the private evaluation set will
almost certainly match it:
```
<data_dir>\train\<class_name>\videos\*.mp4  (+ videos.csv, ground_truth.csv)
<data_dir>\test\videos\*.mp4                (+ videos.csv, ground_truth.csv)
```
Every script here takes `--data_dir`, so a new drop is a one-flag swap —
**do not hardcode a path**. If `test\ground_truth.csv` is absent (a genuinely
private set), everything still runs; you simply cannot self-score, only
produce `predictions.csv` and submit it blind.

The 12 official class strings (must match exactly, enforced by
`label_map.validate_label`):
```
normal, traffic_accident, traffic_congestion, stalled_or_broken_down_vehicle,
vehicle_blocking_traffic, wrong_way_driving, road_spill_or_debris,
waterlogging_or_flood, fire, smoke, fighting_or_violence,
loitering_or_suspicious_presence
```

## 3. The winning pipeline, end to end

This is what actually produces score. Five steps, in order:

```
:: 1. Train the appearance classifier (LOCAL, GTX 1650, ~15-20 min for 12
::    epochs; extraction/caching is cached so a rerun is much faster). This is
::    the thing that catches fire/smoke/flood/debris/fighting/loitering/
::    wrong-way/congestion/blocking - conditions a motion tracker structurally
::    cannot see because they are visible in a SINGLE frame, not defined by
::    movement. Rules-only scored F1 0.0 on every one of these.
%PY% src\train_appearance.py --data_dir <data_dir> --epochs 12

:: 2. Dump per-video class probabilities on the test split
%PY% src\appearance_classifier.py --data_dir <data_dir> --split test ^
  --weights C:\dvad\models\appearance11.pt --dump C:\dvad\outputs\app_scores.json

:: 3. Pick the decision rule by sweeping against the REAL scorer (not a guess -
::    the threshold was guessed twice before and missed twice; an 11-class head
::    spreads probability mass differently than a 7-class one did)
%PY% src\tune_appearance.py --scores C:\dvad\outputs\app_scores.json ^
  --gt <data_dir>\test\ground_truth.csv --write C:\dvad\outputs\predictions.csv

:: 4. Score it locally (only works if test\ground_truth.csv exists - the
::    organisers ship it on the PUBLIC set specifically so this step is possible
::    before the private submission)
%PY% src\score_submission.py --gt <data_dir>\test\ground_truth.csv ^
  --pred C:\dvad\outputs\predictions.csv

:: 5. predictions.csv is the submission. Sanity-check the row count matches
::    the number of test videos (a missing video_id scores worse than a wrong
::    guess - both run_ahc_dataset.py and tune_appearance.py already backfill
::    a `normal` row for any video with no score, keep it that way).
```

**Do not run other heavy GPU/CPU jobs alongside step 1.** Measured: free RAM
hit 1.4GB of 7.5GB and epochs slowed ~5x from contention when a scoring job ran
concurrently. It is CPU-bound on JPEG decode (`num_workers=0`, deliberate for
an 8GB machine).

### If a private/day-of test set arrives with NO `ground_truth.csv`
Steps 1–3 are unchanged (`--data_dir` still points at wherever `train/` lives —
reuse the checkpoint from the day's earlier training run, no need to retrain
unless the class folders changed). Skip step 4 (nothing to score against) and
submit `predictions.csv` from step 3 directly. If time allows, use the
DECISION RULE already validated on the public test set (see §5) rather than
re-tuning blind on a set with no way to check the result — an untested rule
picked to look good on a training-adjacent proxy is a worse bet than a rule
already measured to generalise once.

## 4. Recovering the one class the classifier deliberately does NOT own

`stalled_or_broken_down_vehicle` is excluded from the classifier's positive
classes on purpose — only 4 training videos, too few to model, and it is the
motion rules' single most reliable true positive (a track stationary in a live
lane for `--stop-seconds`, unconfounded with parking/shoulder by zone
calibration). To recover it without touching the classifier:
```
%PY% src\run_ahc_dataset.py --data_dir <data_dir> --split test ^
  --label-source hybrid --appearance-weights C:\dvad\models\appearance11.pt ^
  --appearance-threshold 0.15 --out C:\dvad\outputs\pred_hybrid.csv
%PY% src\score_submission.py --gt <data_dir>\test\ground_truth.csv --pred C:\dvad\outputs\pred_hybrid.csv
```
This costs ~20–40s per video (the full motion pipeline runs alongside the
~0.3s classifier call) to recover one class on one video. **Compare the
resulting macro-F1 against step 3's plain-classifier result and keep whichever
actually scores higher** — do not assume the hybrid path wins just because it
does more work. As of last measurement the classifier-only path is proven; the
hybrid path was written but its net effect on the score had not yet been
checked end to end. Verify before trusting it in a demo.

## 5. Findings already measured — do not re-litigate these under time pressure

- **Per-class decision thresholds fitted on the 34-video PUBLIC TEST SET
  overfit it.** `tune_appearance.py --per-class` lifts in-sample macro-F1 to
  0.289, but its own `--cv` (leave-one-video-out) check returns 0.230 — *worse*
  than the plain global rule's ~0.245–0.256. Reproduce:
  `tune_appearance.py --scores ... --gt ... --per-class --cv`. **Use the global
  rule from step 3, not per-class thresholds tuned on test.**
- **Per-class thresholds calibrated on the classifier's own held-out VAL split
  (365 videos, never seen by the model) ALSO fail to transfer to test** —
  measured macro-F1 **0.126**, worse than either of the above. This was tried
  specifically because the val-overfitting explanation above implied a bigger,
  genuinely-held-out calibration set should fix it; it didn't. The likely cause
  is real domain shift, not sample size: the organisers explicitly separate
  training-pool sources from the reserved test-set source at the video level,
  so val (drawn from the training pool) is not a faithful proxy for test's
  camera/scene distribution. Reproduce: `calibrate_thresholds.py --apply
  C:\dvad\outputs\app_scores.json --gt <data_dir>\test\ground_truth.csv`.
  **Conclusion: use the global rule. Per-class thresholds have failed twice,
  fitted two different ways — do not try a third variant on the day.**
- **Higher validation accuracy during training does not reliably mean a higher
  test score**, for the same domain-shift reason. Measured: val macro-recall
  climbed 0.650 → 0.742 across epochs 1→9, while test macro-F1 measured at an
  early checkpoint (0.256) was *higher* than at a later, better-val checkpoint
  (0.188 at epoch 6). **`train_appearance.py` currently keeps only the single
  best-by-val-recall checkpoint, overwriting earlier ones** — so if you retrain
  during the event, you cannot get back an earlier checkpoint that might score
  better on test once it's gone. Consider running step 2–4 against a mid-training
  checkpoint copy, not only the final one, before committing to a submission.
  If retraining, copy `appearance11.pt` aside after a few early epochs as
  insurance: `copy C:\dvad\models\appearance11.pt C:\dvad\models\appearance11_epochN.pt`.
- **Speed-based congestion detection cannot work on this footage** — measured
  with `diag_speeds.py`, a genuinely `normal` clip reads as MORE congested than
  both ground-truth congestion clips at every speed threshold from 0.05 to
  0.50. Box jitter on a few-pixel vehicle at low resolution swamps the speed
  estimate; no threshold separates them. This is why `traffic_congestion` moved
  to the appearance classifier instead of the motion rule.
- **The test set is three datasets in one costume**: some clips are 5.7–26s,
  some are 2fps with 30–706 total frames (track association and speed are pure
  noise at 0.5s between frames), some are 240–629s. Six of 34 videos are
  structurally hostile to any motion-based rule — one more reason the
  appearance classifier, which only needs a handful of representative frames,
  carries more of the score than the motion pipeline on this dataset.
- **T030** is listed in `videos.csv`/`ground_truth.csv` but the video file is
  missing from the public pack. Every submission-writer already backfills a
  `normal` row for it — verify this still holds if the day-of set differs.

## 6. Drone/night footage specifics (motion-rules path, still valid)

```
:: aerial + fine-tuned weights (4.3x recall over stock COCO defaults, measured
:: on VisDrone: 0.661 vs 0.152 overall recall)
%PY% src\pipeline.py --source <clip> --zones <zones> --decision rules ^
  --aerial --weights C:\dvad\models\yolo26n_visdrone.pt

:: night: --night alone (lowers conf to 0.20) recovers ABOVE-daylight detection
:: counts. Do NOT reach for CLAHE - measured 23.7->4.0fps and FEWER detections.
%PY% src\pipeline.py --source <night_clip> --zones <zones> --decision rules --night
```
If the pipeline looks broken on unfamiliar footage, sanity-check Stage 1 alone
before touching thresholds — the fault is almost always detection, not the
anomaly logic:
```
%PY% src\detect_track.py --source <clip> --imgsz 1280 --conf 0.1 --save C:\dvad\outputs\check.mp4
```

## 7. Demo script

1. **The classifier's actual numbers on the real public test set** — lead with
   this, it's the one number that's genuinely measured against organiser data,
   not self-graded synthetic footage: macro-F1 (quote whatever step 4 measures
   TODAY, with the checkpoint it came from — do not quote last night's number
   from memory).
2. **The architecture story**: three tiers doing three different jobs — a
   motion tracker for things defined by HOW something moves (a stopped
   vehicle, wrong-way driving), an always-on classifier for things that are
   simply VISIBLE in one frame (fire, smoke, flooding, debris — measured to be
   literally unreachable by motion rules, F1 0.0 on all four), and the VLM
   reserved for open-vocabulary hazards neither can name in advance. All of it
   runs on a 4GB GTX 1650 with no VLM in the runtime hot path for a normal
   classification pass — the VLM is optional enrichment.
3. **The honest engineering finding on the VLM boolean**: a 3B VLM cannot
   reliably decide "is this anomalous" from a single still frame, because a
   still frame contains no motion — measured at chance (3/6) across four
   prompt revisions on the injected stopped-truck clip, while its *prose* was
   consistently accurate. So the deterministic tracker owns the boolean; the
   VLM only observes and may escalate, never silently clear. Full detail in
   `CLAUDE.md`.
4. **The annotated video** (motion-rules path) as the visual — red box + alert
   banner on a stopped vehicle, green tracked boxes with dwell timers on
   everything else:
   ```
   %PY% src\demo.py --source <clip>
   ```

## 8. If the wifi dies

Nothing in the demo path needs the network, tested with all external traffic
routed to an unroutable address: `--decision rules --aerial` and
`--decision hybrid --backend ollama` both completed correctly.
- YOLO weights and the appearance classifier checkpoint are cached locally in
  `C:\dvad\models\`.
- Ollama serves on `localhost:11434` — loopback, works with wifi down.
- `--decision rules` needs no model call at all; `--backend mock` runs the
  whole pipeline with zero weights.
- Steps 1–4 above (training, dumping, tuning, scoring) are all 100% local and
  need no network whatsoever.

## 9. Kaggle / VLM LoRA fine-tune — optional stretch, currently blocked

`build_vlm_dataset.py` converts the organisers' `description_summary` captions
plus the frame cache into Unsloth-ready `train.jsonl`/`val.jsonl`. It is
written and has never been run end to end. **It cannot start until Kaggle
Settings → Phone Verification is completed** (no GPU accelerator without it).
Use the **T4**, never the P100 (`machine_shape: "NvidiaTeslaT4"` — the default
P100 dies at `get_peft_model` with current bitsandbytes/Unsloth kernels).

Only attempt this if steps 1–4 above are done, scored, and submitted first —
the local classifier already carries the score, and a fine-tune landing after
venue wifi degrades cannot be relied on for the demo.

```
:: one-time setup if not already done
%PY% src\setup_kaggle.py --token KGAT_xxxxxxxx
%PY% src\setup_kaggle.py --verify-only

:: build the VLM dataset from real organiser captions (once phone-verified)
%PY% src\build_vlm_dataset.py --data_dir <data_dir> --out C:\dvad\data\vlm_dataset

:: run the fine-tune from the CLI, no browser clicks
%PY% src\push_notebook.py --push --slug dvad-finetune-qwen2-5-vl --title "dvad finetune qwen2 5 vl"
%PY% src\push_notebook.py --wait  --slug dvad-finetune-qwen2-5-vl
%PY% src\push_notebook.py --pull  --slug dvad-finetune-qwen2-5-vl
```
Two Kaggle traps already hit and fixed — do not reintroduce them:
- **The URL slug comes from the TITLE, not the id.** Keep `--slug` and
  `--title` consistent or `--status` 404s on a kernel that exists.
- **nbformat needs a trailing `\n` on every source line**, or Kaggle joins a
  cell onto one line and it silently breaks. `notebooks\build_notebook.py`
  regenerates the notebook and asserts this — if you edit it, regenerate,
  don't hand-edit the JSON.
