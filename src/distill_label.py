"""STAGE 6 - distillation: label event frames with a large teacher VLM.

Providers: groq (free, 14,400 req/day - the default), openrouter (free but only
50 req/day on a zero balance), anthropic (paid). Groq and OpenRouter share one
OpenAI-compatible code path.

    python src\\distill_label.py --provider groq --list-models
    python src\\distill_label.py --events ...\\harvest_events.jsonl --provider groq


The teacher is shown the *same* highlighted frame and the *same* context string
that Stage 3 sees at inference time, and answers in the *same* schema. The
student therefore learns exactly the mapping it will be asked to perform.

    # preferred: label the events a pipeline run actually triggered
    python src\\distill_label.py --events C:\\dvad\\outputs\\events_vehicles.jsonl --limit 40

    # or label a bare folder of frames
    python src\\distill_label.py --data_dir C:\\dvad\\outputs\\events --limit 40

    python src\\distill_label.py --events ... --dry-run   # cost estimate, no API calls

Resumable: rows already present in the output are skipped, so Ctrl+C is safe.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from common import AnomalyVerdict, DATA_DIR, OUTPUTS_DIR, read_jsonl
from vlm_reason import downscale, highlight_target, parse_verdict

# Per-MTok rates for the cost estimate. Update if pricing changes.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Groq and OpenRouter both speak the OpenAI chat-completions dialect, so one
# code path covers both - only the base URL, key and default model differ.
PROVIDERS = {
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5",
        "free": False,
    },
    "groq": {
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1",
        # Verified by probe on 2026-09-04: of the 14 models Groq serves this
        # account, only qwen3.8-27b and qwen3.6-27b accept images. The widely
        # documented llama-4-scout is NOT available. qwen3.8 answers cleanly;
        # qwen3.6 leaks <think> tags. Re-check with --probe-vision if this drifts.
        "default_model": "qwen/qwen3.8-27b",
        "free": True,
        # The binding limit is TOKENS per minute (8000 measured on qwen3.8-27b),
        # not requests, and an image costs far more tokens than text. Parallel
        # requests just trade 200s for 429s, so keep this at 1.
        "concurrency_cap": 1,
        "max_side": 512,  # 1024px images blew the TPM budget; 512 is ample here
        "signup": "https://console.groq.com/keys",
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-4-31b-it:free",
        "free": True,
        # Only 50 requests/day on a $0 balance - keep concurrency low so a
        # burst does not burn the daily quota on retries.
        "concurrency_cap": 2,
        "signup": "https://openrouter.ai/keys",
    },
}

TEACHER_SYSTEM = (
    "You are an expert aerial-surveillance analyst producing ground-truth labels to train a "
    "smaller on-device model. You will see a drone video frame with one object outlined in a "
    "bright magenta box, plus tracker context about that object.\n\n"
    "Decide whether the highlighted object represents a genuine anomaly that a human operator "
    "must act on. Judge it in context:\n"
    "- A vehicle stopped in a parking area or a designated shoulder is normal.\n"
    "- A vehicle stopped in a live traffic lane is an incident.\n"
    "- If most surrounding vehicles are also stopped, this is congestion, not an incident.\n"
    "- A pedestrian on a highway carriageway is a serious anomaly.\n"
    "- Smoke, fire, collision debris, or a crowd forming around a stopped vehicle raise severity.\n\n"
    "severity scale: 0.0-0.2 benign, 0.3-0.5 worth logging, 0.6-0.8 dispatch an operator, "
    "0.9-1.0 emergency.\n"
    "Be decisive and calibrated - these labels become training targets. Keep `reason` to one "
    "short sentence citing what you actually see in the frame.\n\n"
    # The literal word "json" is REQUIRED here: Groq rejects response_format
    # json_object with a 400 unless it appears somewhere in the messages. It also
    # stops models returning markdown prose like "**Label:** Genuine Anomaly".
    'Reply with a single json object and nothing else, exactly these three keys:\n'
    '{"anomalous": true or false, "severity": 0.0 to 1.0, "reason": "one short sentence"}'
)

_print_lock = threading.Lock()


def build_prompt(context: str) -> str:
    return (
        f"Tracker context for the highlighted object:\n{context}\n\n"
        "Label this object: is it a genuine anomaly, how severe, and why?"
    )


def load_tasks(args) -> list[dict]:
    """Build the work list from either an events jsonl or a frames directory."""
    tasks: list[dict] = []

    if args.events:
        rows = read_jsonl(Path(args.events))
        frames_dir = Path(args.frames_dir) if args.frames_dir else OUTPUTS_DIR / "events"
        for r in rows:
            fp = frames_dir / r["frame_file"] if r.get("frame_file") else None
            if fp is None or not fp.exists():
                continue
            tasks.append(
                {
                    "image_path": fp,
                    "context": r["context"],
                    "bbox": tuple(r["bbox"]) if r.get("bbox") else None,
                    "meta": {
                        "source_video": r.get("video"),
                        "frame_idx": r.get("frame_idx"),
                        "track_id": r.get("track_id"),
                        "event_kind": r.get("kind"),
                        "zone_kind": r.get("zone_kind"),
                        "features": r.get("features"),
                    },
                }
            )
    else:
        d = Path(args.data_dir)
        images = sorted(
            p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for p in images:
            tasks.append(
                {
                    "image_path": p,
                    "context": args.fallback_context,
                    "bbox": None,
                    "meta": {"source_video": None, "frame_idx": None, "track_id": None},
                }
            )

    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def encode_image(path: Path, bbox, max_side: int) -> tuple[str, str]:
    """Return (base64 jpeg, media_type) of the highlighted, downscaled frame."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    prepared = downscale(highlight_target(frame, bbox), max_side)
    ok, buf = cv2.imencode(".jpg", prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError(f"jpeg encode failed for {path}")
    return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"


def _row(task: dict, verdict: AnomalyVerdict, model: str, usage: dict) -> dict:
    return {
        "image": task["image_path"].name,
        "context": task["context"],
        "anomalous": verdict.anomalous,
        "severity": verdict.severity,
        "reason": verdict.reason,
        "teacher_model": model,
        "usage": usage,
        **{k: v for k, v in task["meta"].items() if v is not None},
    }


def label_one_anthropic(client, task: dict, args) -> dict:
    import anthropic

    image_b64, media_type = encode_image(task["image_path"], task["bbox"], args.max_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": build_prompt(task["context"])},
            ],
        }
    ]

    last_err: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            resp = client.messages.parse(
                model=args.model,
                max_tokens=1024,
                system=TEACHER_SYSTEM,
                messages=messages,
                output_format=AnomalyVerdict,
            )
            # A safety decline on one frame must not abort a long labeling run.
            if getattr(resp, "stop_reason", None) == "refusal":
                return {"error": "refusal", "image": task["image_path"].name}
            verdict: AnomalyVerdict = resp.parsed_output
            return _row(task, verdict, args.model,
                        {"input_tokens": resp.usage.input_tokens,
                         "output_tokens": resp.usage.output_tokens})
        except anthropic.BadRequestError as e:
            return {"error": f"bad_request: {e}", "image": task["image_path"].name}
        except anthropic.AuthenticationError:
            raise
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_err = e
            if attempt < args.retries:
                time.sleep(2.0 * (attempt + 1))
    return {"error": f"failed after retries: {last_err}", "image": task["image_path"].name}


def label_one_openai_compatible(task: dict, args) -> dict:
    """Groq and OpenRouter: same OpenAI chat-completions dialect, vision via data URI."""
    import requests

    cfg = PROVIDERS[args.provider]
    image_b64, media_type = encode_image(task["image_path"], task["bbox"], args.max_side)
    payload = {
        "model": args.model,
        "temperature": 0.1,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                    {"type": "text", "text": build_prompt(task["context"])},
                ],
            },
        ],
    }
    if not args.no_json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    if args.provider == "openrouter":
        # OpenRouter asks for these for attribution; harmless elsewhere.
        headers["HTTP-Referer"] = "https://github.com/dvad-hackathon"
        headers["X-Title"] = "DVAD drone anomaly distillation"

    last_err = None
    for attempt in range(args.retries + 1):
        try:
            r = requests.post(f"{cfg['url']}/chat/completions", json=payload,
                              headers=headers, timeout=args.timeout)
            if r.status_code == 429:
                # Groq free tier is capped on TOKENS per minute (8000 on
                # qwen3.8-27b), not just requests, and images are token-heavy.
                wait = float(r.headers.get("retry-after", 8 * (attempt + 1)))
                last_err = f"429 rate limited (waited {wait:.0f}s)"
                time.sleep(min(wait, 45))
                continue
            if r.status_code == 401:
                raise PermissionError(f"{args.provider} rejected the API key (401)")
            if r.status_code >= 400:
                # Surface the provider's actual message - a bare
                # "400 Bad Request" hid a one-line, trivially fixable cause.
                try:
                    detail = r.json().get("error", {}).get("message", r.text)
                except Exception:  # noqa: BLE001
                    detail = r.text
                return {"error": f"{r.status_code}: {str(detail)[:300]}",
                        "image": task["image_path"].name}
            body = r.json()
            text = body["choices"][0]["message"]["content"]
            verdict = parse_verdict(text)  # tolerant: salvages JSON out of prose
            usage = body.get("usage") or {}
            return _row(task, verdict, args.model,
                        {"input_tokens": usage.get("prompt_tokens", 0),
                         "output_tokens": usage.get("completion_tokens", 0)})
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the run
            last_err = f"{type(e).__name__}: {e}"
            if attempt < args.retries:
                time.sleep(2.0 * (attempt + 1))
    return {"error": f"failed after retries: {last_err}", "image": task["image_path"].name}


def _fetch_models(args) -> list[dict]:
    import requests

    cfg = PROVIDERS[args.provider]
    r = requests.get(f"{cfg['url']}/models",
                     headers={"Authorization": f"Bearer {args.api_key}"}, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def list_models(args) -> None:
    """Print the provider's live model list - free-tier model IDs drift."""
    cfg = PROVIDERS[args.provider]
    if args.provider == "anthropic":
        print("Use the Anthropic docs for model ids; the default is claude-opus-5.")
        return
    models = _fetch_models(args)
    print(f"{len(models)} model(s) on {args.provider}:\n")
    for m in models:
        mid = m.get("id", "?")
        arch = m.get("architecture") or {}
        modalities = [str(x).lower() for x in (arch.get("input_modalities") or [])]
        vision = "image" in modalities or any(
            k in mid.lower() for k in ("vl", "vision", "scout", "maverick", "gemma-4", "omni")
        )
        free = ":free" in mid or m.get("pricing", {}).get("prompt") in ("0", 0, "0.0")
        tags = ("vision? " if vision else "        ") + ("FREE" if free else "")
        print(f"  {tags:<14} {mid}")
    print("\nName-based guessing is unreliable - Groq's qwen3.8-27b has vision but "
          "says so nowhere in its id.\nRun --probe-vision to test them for real.")


def probe_vision(args) -> None:
    """Send a tiny image to each model and see which actually accept it.

    Provider docs and model names both lied about this, so the only reliable
    answer is an empirical one. Costs a handful of trivial free-tier calls.
    """
    import base64

    import cv2
    import numpy as np
    import requests

    cfg = PROVIDERS[args.provider]
    img = np.full((64, 96, 3), 40, np.uint8)
    cv2.rectangle(img, (20, 20), (60, 45), (200, 60, 60), -1)
    ok, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf.tobytes()).decode()

    skip = ("whisper", "orpheus", "prompt-guard", "safeguard", "tts", "embed", "allam")
    ids = [m["id"] for m in _fetch_models(args)
           if not any(s in m["id"].lower() for s in skip)]
    print(f"probing {len(ids)} candidate(s) on {args.provider} for image support\n")

    working = []
    for mid in ids:
        payload = {
            "model": mid, "max_tokens": 40,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What colour is the rectangle? One word."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        }
        try:
            r = requests.post(f"{cfg['url']}/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {args.api_key}"}, timeout=120)
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
                thinks = "<think>" in txt.lower()
                print(f"  VISION OK  {mid:<40} -> {txt[:60]}"
                      + ("   [emits <think> tags]" if thinks else ""))
                working.append(mid)
            else:
                msg = r.json().get("error", {}).get("message", r.text)[:70].replace("\n", " ")
                print(f"  no image   {mid:<40} -> {r.status_code} {msg}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR      {mid:<40} -> {type(e).__name__}")
    print(f"\n{len(working)} model(s) accept images. Use one with --model.")
    if working:
        print(f"  e.g. --model {working[0]}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Label event frames with a teacher VLM (Groq / OpenRouter / Anthropic)."
    )
    src = p.add_mutually_exclusive_group(required=False)  # not needed for --list-models
    src.add_argument("--events", help="Events .jsonl from pipeline.py (preferred - carries context).")
    src.add_argument("--data_dir", help="A folder of frames to label.")
    p.add_argument("--frames-dir", default=None, help="Where --events frame_file entries live.")
    p.add_argument("--out", default=None, help="Output jsonl (default: <data>/pseudo_labels.jsonl).")
    p.add_argument("--provider", default="groq", choices=sorted(PROVIDERS),
                   help="Teacher provider. groq (default) is free: 14,400 req/day. "
                        "openrouter is free but only 50 req/day on a zero balance.")
    p.add_argument("--model", default=None, help="Teacher model id (default: per provider).")
    p.add_argument("--list-models", action="store_true",
                   help="Print the provider's live model list and exit (free ids drift).")
    p.add_argument("--probe-vision", action="store_true",
                   help="Empirically test which models accept images. Model names and "
                        "provider docs both lied about this - trust only the probe.")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--no-json-mode", action="store_true",
                   help="Disable response_format=json_object if a model rejects it.")
    p.add_argument("--limit", type=int, default=0, help="Label at most N frames (0 = all).")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-side", type=int, default=1024, help="Downscale frames to this longest side.")
    p.add_argument(
        "--fallback-context",
        default="An object of interest is highlighted in this aerial frame. No tracker context available.",
    )
    p.add_argument("--dry-run", action="store_true", help="Estimate cost and exit without calling the API.")
    p.add_argument("--overwrite", action="store_true", help="Ignore existing rows instead of resuming.")
    args = p.parse_args()

    cfg = PROVIDERS[args.provider]
    args.model = args.model or cfg["default_model"]
    args.api_key = os.environ.get(cfg["env"], "")
    if not args.api_key and not args.dry_run:
        raise SystemExit(
            f"\n{cfg['env']} is not set.\n"
            f"  1. Get a free key: {cfg.get('signup', 'https://console.anthropic.com')}\n"
            f'  2. setx {cfg["env"]} "your-key-here"\n'
            f"  3. Open a NEW terminal (setx only affects new shells), then re-run.\n"
        )
    args.concurrency = min(args.concurrency, cfg.get("concurrency_cap", 8))
    # Providers with tight token budgets get a smaller default image, unless the
    # user asked for a specific size.
    if "max_side" in cfg and args.max_side == 1024:
        args.max_side = cfg["max_side"]

    if args.list_models:
        list_models(args)
        return
    if args.probe_vision:
        probe_vision(args)
        return
    if not args.events and not args.data_dir:
        raise SystemExit("Give one of --events or --data_dir (or --list-models).")

    out_path = Path(args.out) if args.out else DATA_DIR / "pseudo_labels.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No frames to label. Run pipeline.py first to generate events.")

    done: set[str] = set()
    if out_path.exists():
        if args.overwrite:
            # Must truncate, not just ignore: the sink opens in append mode, so
            # re-labelling without truncating left stale rows behind and the
            # class balance was computed over both label sets.
            out_path.unlink()
            print("[overwrite] previous labels discarded.")
        else:
            done = {r.get("image") for r in read_jsonl(out_path) if "error" not in r}
            if done:
                print(f"[resume] {len(done)} frame(s) already labeled - skipping those.")
    todo = [t for t in tasks if t["image_path"].name not in done]

    # ~1300 input tokens/frame at 1024px + ~90 output tokens is a good estimate.
    in_rate, out_rate = PRICING.get(args.model, (5.00, 25.00))
    est = 0.0 if cfg["free"] else len(todo) * (1300 * in_rate + 90 * out_rate) / 1_000_000
    print(f"[plan] provider : {args.provider}  ({'FREE tier' if cfg['free'] else 'paid'})")
    print(f"[plan] model    : {args.model}")
    print(f"[plan] frames   : {len(todo)} to label, concurrency {args.concurrency}")
    print(f"[plan] cost     : {'$0.00 (free tier)' if cfg['free'] else f'~${est:.3f}'}")
    print(f"[plan] output   : {out_path}")

    if args.dry_run:
        print("[dry-run] no API calls made.")
        return
    if not todo:
        print("Nothing to do.")
        return

    client = None
    if args.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        worker = lambda t: label_one_anthropic(client, t, args)  # noqa: E731
    else:
        worker = lambda t: label_one_openai_compatible(t, args)  # noqa: E731

    written = errors = 0
    in_tok = out_tok = 0
    t0 = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(worker, t): t for t in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                with _print_lock:
                    if "error" in row:
                        errors += 1
                        print(f"  [{i}/{len(todo)}] ERROR {row['image']}: {row['error']}")
                    else:
                        written += 1
                        in_tok += row["usage"]["input_tokens"]
                        out_tok += row["usage"]["output_tokens"]
                        flag = "ANOMALY" if row["anomalous"] else "benign "
                        print(
                            f"  [{i}/{len(todo)}] {flag} sev={row['severity']:.2f} "
                            f"{row['image'][:44]:<44} {row['reason'][:60]}"
                        )

    actual = 0.0 if cfg["free"] else (in_tok * in_rate + out_tok * out_rate) / 1_000_000
    print("\n=== distillation summary ===")
    print(f"labeled        : {written}")
    print(f"errors         : {errors}")
    print(f"tokens in/out  : {in_tok} / {out_tok}")
    print(f"actual cost    : {'$0.00 (free tier)' if cfg['free'] else f'${actual:.4f}'}")
    if written:
        print(f"rate           : {(time.perf_counter() - t0) / written:.1f}s per frame "
              f"(free-tier token limit, not model speed)")
    print(f"elapsed        : {time.perf_counter() - t0:.1f}s")
    print(f"output         : {out_path}")

    rows = [r for r in read_jsonl(out_path) if "error" not in r]
    if rows:
        pos = sum(1 for r in rows if r["anomalous"])
        print(f"class balance  : {pos} anomalous / {len(rows) - pos} benign")
        if pos == 0 or pos == len(rows):
            print("[warn] single-class labels - the student cannot learn a boundary. "
                  "Label more varied frames (lower --stop-seconds, or add calm footage).")


if __name__ == "__main__":
    main()
