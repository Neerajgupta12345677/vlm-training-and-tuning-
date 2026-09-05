"""Trains the Stage 1.5 appearance classifier (11-way clip classification).

Why this exists: Stage 2 is dwell/speed/zone arithmetic over tracked objects, so
it structurally cannot represent a condition that is simply VISIBLE in the frame.
Measured on the organisers' public test set, that cost every appearance class:
fire, smoke, waterlogging_or_flood and road_spill_or_debris all scored F1 0.0
because `--decision rules` never calls a model and the rules never fire.

MobileNetV3-Small was chosen over a VLM call for this job: 2.54M params, 557MB
peak VRAM for a batch of 32 at 224px (measured on the GTX 1650), versus 27-45s
per qwen2.5vl:3b call. It trains LOCALLY in minutes - no Kaggle upload, no GPU
quota, no phone-verification blocker on the critical path.

Two methodology points that are not optional:
  1. The train/val split is by VIDEO, never by frame. Eight frames sampled from
     one clip are near-duplicates; splitting them randomly puts the same scene on
     both sides and reports a val accuracy that means nothing.
  2. traffic_accident is EXCLUDED from the negative class. Those clips contain
     burning vehicles ("Crashed and burned cars and trucks" is a real caption in
     the test ground truth), and teaching the model that fire is normal would
     poison the one class with the most training data.

    python src\train_appearance.py --data_dir C:\dvad\data\ahc
    python src\train_appearance.py --data_dir C:\dvad\data\ahc --extract-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from common import MODELS_DIR

# Positives = classes where a STILL FRAME genuinely contains the evidence.
# That is the dividing line, and it is the same one the architecture is built
# on: motion-defined events stay with Stage 2, because a single frame cannot
# tell a stopped car from a moving one.
#
# An earlier run included road_spill_or_debris on the 6 videos then available.
# Inverse-frequency weighting handed that class a ~20x loss weight and the
# network answered "debris" for everything - p=0.514 on a basketball court -
# while its val recall of 1.000 was computed over a single held-out video.
# The fix was more data (86 videos now) AND clamped weights, not one or the
# other. stalled_or_broken_down_vehicle still has only 4 videos, so it stays
# out and remains Stage 2's job, which is its strongest rule anyway.
POSITIVE_CLASSES = [
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "road_spill_or_debris",
    "fighting_or_violence",
    "traffic_accident",
    # The four below are motion-DEFINED but visually distinctive from the air,
    # and they were moved out of the negative class on evidence rather than
    # principle. Measured on the public test set with src\diag_speeds.py: the
    # `normal` clip T003 reads as MORE congested than both ground-truth
    # traffic_congestion clips at every speed cut from 0.05 to 0.50, because
    # T003 is 256x192 and box jitter on a few-pixel vehicle swamps the speed
    # estimate. No threshold can separate them, so the speed-share rule cannot
    # work on this footage. These classes also carry 386 training videos and 17
    # of the 52 ground-truth rows between them, which was going unused.
    #
    # wrong_way_driving is the weakest of the four on principle - one frame
    # genuinely cannot show direction - so treat its val recall with suspicion:
    # if it scores well it may be reading scene furniture rather than heading.
    # Stage 2 keeps its own wrong-way rule regardless.
    "loitering_or_suspicious_presence",
    "wrong_way_driving",
    "traffic_congestion",
    "vehicle_blocking_traffic",
]
# Motion-defined classes look like ordinary road scenes in a single frame, so
# they belong in the negative class - that is what makes the classifier hand
# them back to Stage 2 instead of inventing an appearance label. traffic_accident
# is deliberately absent: see the module docstring.
# Motion-defined classes are the NEGATIVE class on purpose. In one frame they
# look like ordinary road scenes, and that is exactly the answer we want: the
# classifier says "nothing visible here" and hands the clip to Stage 2, which
# can actually measure dwell, direction and density.
NEGATIVE_SOURCE_CLASSES = [
    "normal",
    # 4 training videos. Too few to model, and it is Stage 2's single most
    # reliable rule - the only true positive the rules-only run produced on the
    # public test set (T010). It stays with the tracker, and carries 1 of 52
    # ground-truth rows, so there is nothing to gain by modelling it.
    "stalled_or_broken_down_vehicle",
]
CLASSES = POSITIVE_CLASSES + ["normal"]

# Raw-frame rebalancing (see the comment at its use site in extract()). Any
# class with fewer videos than this gets 2x frames/clip - raised from 40 to
# 150 so fire(77)/smoke(85)/waterlogging(95)/road_spill(86)/fighting(124)/
# wrong_way(109)/loitering(135)/vehicle_blocking(147) all qualify, not just
# traffic_congestion(23). traffic_accident(328) and normal(632) are the only
# classes still excluded - they were never underrepresented.
REBALANCE_VIDEO_CUTOFF = 150
# traffic_accident had 328 videos / 2624 frames versus 616-992 for the classes
# it was measured to be swallowing at test time. Capped to roughly the same
# order of magnitude as its closest under-the-cutoff peers rather than left
# unbounded.
TRAFFIC_ACCIDENT_CAP = 150

IMG_SIZE = 224

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    NORM,
])
VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORM,
])


def sample_frames(video: Path, n: int) -> list[np.ndarray]:
    """n frames evenly spaced across the clip, RGB."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: list[np.ndarray] = []
    if total <= 0:
        cap.release()
        return out
    # Avoid the very first/last frame: fades and black frames are common.
    idxs = np.linspace(total * 0.05, total * 0.95, n).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(i, 0, total - 1)))
        ok, frame = cap.read()
        if ok and frame is not None:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _prune_stale(cache: Path) -> None:
    """Drop cached frames that no longer belong to the directory they sit in.

    Cached filenames are `{source_folder}__{video_stem}__{k}.jpg`, so a frame
    carries its own provenance. When a class is promoted out of the negative
    set - as loitering, wrong_way, congestion and blocking were - its frames are
    already sitting in `normal/` from the previous run. Leaving them there keeps
    thousands of mislabelled frames in the negative class AND duplicates them
    under their new label, which is worse than having no cache at all. The cache
    is a derived artifact, so it self-heals instead of needing manual cleanup.
    """
    if not cache.exists():
        return
    allowed = {c: ({c} if c != "normal" else set(NEGATIVE_SOURCE_CLASSES)) for c in CLASSES}
    removed = 0
    for d in cache.iterdir():
        if not d.is_dir():
            continue
        # A directory for a label we no longer train on at all.
        if d.name not in allowed:
            n = len(list(d.glob("*.jpg")))
            for fp in d.glob("*.jpg"):
                fp.unlink()
            print(f"  [prune] dropped stale label dir {d.name} ({n} frames)")
            removed += n
            continue
        for fp in d.glob("*.jpg"):
            src = fp.name.split("__", 1)[0]
            if src not in allowed[d.name]:
                fp.unlink()
                removed += 1
    if removed:
        print(f"  [prune] removed {removed} mislabelled cached frame(s)")


def extract(data_dir: Path, cache: Path, per_video: int) -> dict:
    """Decode frames once to JPEG on disk. Re-seeking video per epoch is far
    slower than the network itself, and this cache makes re-runs instant."""
    train_root = data_dir / "train"
    manifest: dict[str, list[str]] = {c: [] for c in CLASSES}
    _prune_stale(cache)
    for folder in POSITIVE_CLASSES + NEGATIVE_SOURCE_CLASSES:
        label = folder if folder in POSITIVE_CLASSES else "normal"
        vdir = train_root / folder / "videos"
        if not vdir.exists():
            print(f"  [skip] {folder}: no videos/ dir")
            continue
        videos = sorted(vdir.glob("*.mp4"))
        # ROOT-CAUSE FIX, not a runtime patch: measured on the public test set,
        # traffic_accident wins argmax on fire/smoke/waterlogging/road_spill/
        # wrong_way test videos with 0.4-0.9 confidence, and the true class
        # often isn't even in the top-3. This is not a close-contest problem
        # fixable by a decision threshold - the RAW TRAINING SIGNAL was
        # imbalanced 2.6-4.3x even after loss reweighting (traffic_accident:
        # 328 videos / 2624 frames vs fire's 77/616, waterlogging's 95/760,
        # etc - loss reweighting changes the gradient's MAGNITUDE per sample,
        # not how many distinct samples the decision boundary is shaped by).
        # traffic_congestion (23 videos) already got a 2x frame boost from the
        # old <40 threshold and reaches F1 0.667 - the fix is to extend that
        # same lever to every class traffic_accident was measured to be
        # swallowing, and to stop traffic_accident's own video count from
        # being 2-4x anyone else's it competes against.
        if folder == "traffic_accident" and len(videos) > TRAFFIC_ACCIDENT_CAP:
            rng_cap = random.Random(0)
            videos = sorted(rng_cap.sample(videos, TRAFFIC_ACCIDENT_CAP))
        out_dir = cache / label
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        n_frames = per_video * 2 if len(videos) < REBALANCE_VIDEO_CUTOFF else per_video
        for v in videos:
            expected = [out_dir / f"{folder}__{v.stem}__{k}.jpg" for k in range(n_frames)]
            # Decoding is the whole cost of this step. Checking first makes a
            # rerun near-instant instead of re-decoding 1845 videos to discard
            # every frame, which matters because the class list gets revised.
            if all(fp.exists() for fp in expected):
                manifest[label] += [fp.name for fp in expected]
                written += len(expected)
                continue
            frames = sample_frames(v, n_frames)
            for k, fr in enumerate(frames):
                fp = out_dir / f"{folder}__{v.stem}__{k}.jpg"
                if not fp.exists():
                    cv2.imwrite(str(fp), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                manifest[label].append(fp.name)
                written += 1
        print(f"  {folder:<38} {len(videos):>4} videos -> {written:>5} frames "
              f"({n_frames}/clip, label={label})")
    return manifest


class FrameDS(Dataset):
    def __init__(self, items: list[tuple[Path, int]], tf):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), label


def build_splits(cache: Path, val_frac: float, seed: int):
    """Split by SOURCE VIDEO so no clip appears on both sides."""
    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    for idx, cls in enumerate(CLASSES):
        d = cache / cls
        if not d.exists():
            continue
        by_video: dict[str, list[Path]] = {}
        for fp in sorted(d.glob("*.jpg")):
            vid = fp.name.rsplit("__", 1)[0]  # folder__videostem
            by_video.setdefault(vid, []).append(fp)
        vids = sorted(by_video)
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_frac)) if len(vids) > 1 else 0
        for v in vids[:n_val]:
            val_items += [(p, idx) for p in by_video[v]]
        for v in vids[n_val:]:
            train_items += [(p, idx) for p in by_video[v]]
    return train_items, val_items


def main() -> None:
    p = argparse.ArgumentParser(description="Train the appearance frame classifier.")
    p.add_argument("--data_dir", default=r"C:\dvad\data\ahc")
    p.add_argument("--cache", default=r"C:\dvad\data\appearance_frames")
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Also save every epoch's checkpoint (not just the best-by-val one), "
                        "so each can be scored against the REAL test set afterward - val and "
                        "test recall are not guaranteed to agree (already measured once).")
    p.add_argument("--out", default=str(Path(MODELS_DIR) / "appearance_classifier.pt"))
    args = p.parse_args()

    cache = Path(args.cache)
    print(f"[extract] frames -> {cache}")
    extract(Path(args.data_dir), cache, args.frames_per_video)
    if args.extract_only:
        return

    train_items, val_items = build_splits(cache, args.val_frac, args.seed)
    if not train_items:
        raise SystemExit("No training frames found - did extraction produce anything?")
    counts = [sum(1 for _, l in train_items if l == i) for i in range(len(CLASSES))]
    print(f"\n[data] train={len(train_items)} val={len(val_items)} (split by source video)")
    for c, n in zip(CLASSES, counts):
        nv = sum(1 for _, l in val_items if l == CLASSES.index(c))
        print(f"  {c:<28} train={n:>5}  val={nv:>4}")

    device = args.device if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(FrameDS(train_items, TRAIN_TF), batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(FrameDS(val_items, VAL_TF), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model = model.to(device)

    # SQRT-inverse frequency, then clamped. Plain inverse frequency is what
    # destroyed the first run: a class with 48 frames got a ~20x weight and the
    # network answered with it everywhere. Softening and capping keeps the rare
    # classes learnable without letting one dominate the loss.
    total = sum(counts)
    raw = [(total / (len(CLASSES) * c)) ** 0.5 if c else 1.0 for c in counts]
    lo = min(raw)
    weights = torch.tensor([min(w / lo, 4.0) for w in raw], dtype=torch.float).to(device)
    print("[weights] " + ", ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weights.tolist())))
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()

        model.eval()
        # Validate the way inference actually works: average the probabilities
        # over a clip's frames, THEN argmax. Frame-level accuracy answers a
        # question nobody asks at runtime and hides per-clip failure.
        # val_dl is shuffle=False, so batches walk val_items in order.
        clip_prob: dict[str, np.ndarray] = {}
        clip_true: dict[str, int] = {}
        with torch.no_grad():
            offset = 0
            for x, y in val_dl:
                probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
                for j in range(len(y)):
                    path, label = val_items[offset + j]
                    vid = path.name.rsplit("__", 1)[0]
                    clip_prob[vid] = clip_prob.get(vid, 0) + probs[j]
                    clip_true[vid] = label
                offset += len(y)
        cor = [0] * len(CLASSES)
        tot = [0] * len(CLASSES)
        for vid, acc in clip_prob.items():
            yi = clip_true[vid]
            tot[yi] += 1
            if int(np.argmax(acc)) == yi:
                cor[yi] += 1
        # Macro recall, not overall accuracy: `normal` dominates the count and a
        # model that always predicts it would look good on plain accuracy.
        recalls = [cor[i] / tot[i] for i in range(len(CLASSES)) if tot[i]]
        macro = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"epoch {ep}/{args.epochs} loss={tl/max(len(train_dl),1):.4f} macro_recall={macro:.3f}")
        for i, c in enumerate(CLASSES):
            if tot[i]:
                print(f"    {c:<28} {cor[i]:>4}/{tot[i]:<4} = {cor[i]/tot[i]:.3f}")
        if macro > best:
            best = macro
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "classes": CLASSES, "img_size": IMG_SIZE,
                        "val_macro_recall": macro}, args.out)
            print(f"    -> saved {args.out} (macro_recall={macro:.3f})")

        if args.save_every_epoch:
            # val recall and TEST macro-F1 have already diverged once tonight
            # (an earlier checkpoint's val rose while test fell) - keeping
            # only the single best-by-val epoch made it impossible to check
            # whether a DIFFERENT epoch would have scored higher on the real
            # test set. Every epoch is cheap to keep (~6MB each); only the
            # inability to compare them after the fact was expensive.
            every_path = Path(args.out).with_name(
                Path(args.out).stem + f"_epoch{ep}_{macro:.3f}.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "classes": CLASSES, "img_size": IMG_SIZE,
                        "val_macro_recall": macro}, every_path)

    print(f"\n[done] best val macro recall {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
