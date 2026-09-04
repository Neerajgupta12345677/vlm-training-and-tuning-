Read CLAUDE.md and PROGRESS.md in this directory fully before doing anything else.

We are setting up a complete ML pipeline from scratch, on a Windows 11
laptop with a GTX 1650 (4GB VRAM, no Tensor Cores) and 8GB RAM, for a
hackathon judged Sept 5, 2026 — that is 2 days away. Training happens on
Kaggle, not locally — the local machine is for pipeline development and
inference only. I am not an expert in every piece of this, so at each major
stage: tell me in plain English what you're about to do and why, run it,
verify it worked with a concrete check, update PROGRESS.md with one short
bullet, then move to the next stage.

Work through these stages IN ORDER. Stop and ask me before any step that
needs a GUI click (WSL2 install, NVIDIA driver, Kaggle phone verification,
Kaggle notebook accelerator selection) — tell me exactly what to do, wait
for my confirmation, then continue. Also stop and ask before starting any
Kaggle GPU session, since it consumes weekly quota (30 hrs/week), and
before installing anything GPU/CUDA-related that's large (>2GB).

STAGE 0 — Environment audit
- Check if WSL2 is installed and has Ubuntu. If not, give me the exact
  PowerShell commands (wsl --install -d Ubuntu), tell me to reboot, wait
  for my confirmation.
- Inside WSL2, verify GPU visibility with `nvidia-smi`. Confirm it reports
  the GTX 1650 4GB. If not visible, explain this usually means the Windows
  NVIDIA driver needs updating.
- Check available RAM inside WSL2 (`free -h`). With only 8GB total, flag
  early if WSL2's memory allocation needs tuning (a `.wslconfig` memory
  cap) so it doesn't starve Windows.

STAGE 1 — Local Python environment (inference-only)
- Install Miniconda inside WSL2 if not present.
- Create conda env `dvad`, Python 3.11.
- Install PyTorch with the CUDA build matching my driver version.
- Verify: torch.cuda.is_available() is True, run a small GPU matmul.

STAGE 2 — Local core libraries
- Install in order, verifying each imports cleanly: transformers, peft,
  ultralytics, supervision, opencv-python, decord, scikit-learn,
  llama-cpp-python (CUDA-enabled build), anthropic SDK.
  Do NOT install vllm locally — not needed here, training is on Kaggle.
- If any install fails, diagnose the actual error before retrying.

STAGE 3 — Kaggle CLI setup
- Install `kaggle` pip package locally.
- Walk me through placing my kaggle.json token in the right location
  (~/.kaggle/kaggle.json inside WSL2, chmod 600).
- Verify with `kaggle datasets list` returning results without error.

STAGE 4 — Model weights (local, for inference)
- Download GGUF quantized Moondream2 and SmolVLM2-500M into models/.
  Confirm each loads and produces one real inference output on a test
  image before considering the download "done." Note the tokens/sec
  observed — this sets realistic expectations for Stage 3 latency later.

STAGE 5 — Pipeline skeleton (local)
- Write src/detect_track.py: YOLO26n (fall back to YOLO11n if it has
  issues) + ByteTrack on a video, prints track IDs/boxes per frame,
  reports achieved FPS.
- Write src/context_state.py: dwell time + placeholder zone label per track.
- Write src/vlm_reason.py: calls local quantized VLM with frame + context
  string, parses structured JSON {anomalous, severity, reason}, with a
  retry-with-simpler-prompt fallback if parsing fails once.
- Wire into src/pipeline.py: runs end-to-end on one placeholder video,
  prints per-track anomaly verdict log.

STAGE 6 — Distillation + Kaggle upload
- Write src/distill_label.py: calls the teacher API on a frames directory,
  saves pseudo-labels to data/pseudo_labels.jsonl in the schema above.
- Run it on placeholder frames now to produce a real (small) sample dataset.
- Write and run the `kaggle datasets create` command to push
  data/pseudo_labels.jsonl as a new private Kaggle Dataset. Confirm it
  appears under my Kaggle account.

STAGE 7 — Kaggle training notebook
- Write notebooks/finetune_kaggle.ipynb: non-interactive, top-to-bottom,
  loads Qwen2.5-VL-3B-Instruct via Unsloth, restricts to one GPU if T4x2 is
  selected (`CUDA_VISIBLE_DEVICES=0`), LoRA config, trains on the attached
  Kaggle Dataset, saves adapter to /kaggle/working/. Tell me explicitly to
  upload this notebook to Kaggle, attach the dataset from Stage 6, select
  the GPU accelerator, and use "Save & Run All (Commit)" so it trains in
  the background. Do this TONIGHT or tomorrow (Sept 3-4) — do not leave
  Kaggle training for Saturday, venue wifi is not reliable.
- Once committed and finished, walk me through downloading the adapter
  output and copying it into models/ locally.

STAGE 8 — Eval harness
- Write src/eval.py: precision/recall/F1 against a labeled placeholder set,
  latency (ms/frame), and a naive feeds-per-GPU estimate from throughput.

After every stage, update PROGRESS.md with one bullet, then tell me
explicitly: what you just did, how you verified it worked, and what stage
comes next — before starting that next stage.
