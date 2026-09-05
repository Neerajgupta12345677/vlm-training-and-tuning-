"""Train the MOTION classifier on ego-compensated temporal-difference frames.

Why a second classifier, rather than more training for the first one
-------------------------------------------------------------------
`train_appearance.py` moved four motion-defined classes into POSITIVE_CLASSES
"on evidence rather than principle", and its own comment flagged the risk:

    wrong_way_driving is the weakest of the four on principle - one frame
    genuinely cannot show direction - so treat its val recall with suspicion:
    if it scores well it may be reading scene furniture rather than heading.

The public leaderboard settled it. Every class in that group failed, and only
that group:

    fighting_or_violence        found 0/3   1 false alarm
    vehicle_blocking_traffic    found 0/2   4 false alarms
    wrong_way_driving           found 0/1   1 false alarm
    loitering                   found 2/7   4 false alarms

No appearance class (fire, smoke, waterlogging, road_spill) appears among the
weakest. Those four account for 10 of the 19 false alarms. It was reading scene
furniture, exactly as predicted.

It could not have done otherwise. Training clips are trimmed to the event -
median event coverage is 100% for 7 of 10 classes, and loitering yields 810
in-event frames against 0 frames of the same scene with nothing happening. So
class correlates perfectly with background, background is the cheapest feature,
and background is constant within a video. That last point is what breaks
Level 3: a per-frame score that does not vary across a clip cannot localise
anything, at any threshold. (Confirmed independently - ranking windows by
confidence put the wrong window first in all 4 L3 videos.)

This model takes the temporal-difference cache from `build_motion_frames.py`
instead. Static scene content cancels to black, so background memorisation is
not available and the network is pushed onto motion - which both defines these
classes and actually varies within a clip.

Two input-specific departures from `train_appearance.py`, neither cosmetic:

  * NO horizontal flip. Mirroring reverses direction of travel, which is the
    entire signal for wrong_way_driving - the standard augmentation would be
    generating labelled counterexamples.
  * NO saturation jitter. The three channels are three time-lags (0.4/1.5/4.0s),
    not colours; scaling them against each other corrupts the relative motion
    magnitudes that distinguish a slow stall from a fast fight.

stalled_or_broken_down_vehicle is excluded: 4 training videos in the whole
dataset. It stays with Stage 2, whose stopped-vehicle rule is its strongest,
and which produced the only true positive of the rules-only run.

    python src\\train_motion.py --cache C:\\dvad\\data\\motion_frames ^
        --out C:\\dvad\\models\\motion_classifier.pt --save-every-epoch
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from train_appearance import FrameDS, NORM, IMG_SIZE, MODELS_DIR

# Motion/duration classes with enough data to model. Ordering is fixed and
# written into the checkpoint, so inference never has to guess it.
CLASSES = [
    "loitering_or_suspicious_presence",
    "wrong_way_driving",
    "traffic_congestion",
    "vehicle_blocking_traffic",
    "fighting_or_violence",
    "traffic_accident",
    "normal",
]

# See the module docstring: no flip (direction is the label for wrong_way), no
# saturation jitter (channels are time-lags, not colours). Brightness/contrast
# stay, mild, so the model tolerates differences in overall scene motion energy
# between a busy junction and a quiet road.
TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORM,
])
VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORM,
])


def build_splits(cache: Path, val_frac: float, seed: int):
    """Split by SOURCE VIDEO so no clip contributes to both sides.

    Frames from one clip are near-duplicates of each other; splitting by frame
    would put those duplicates on both sides and report a val score that is
    mostly memorisation.
    """
    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    for idx, cls in enumerate(CLASSES):
        d = cache / cls
        if not d.exists():
            continue
        by_video: dict[str, list[Path]] = {}
        for fp in sorted(d.glob("*.jpg")):
            by_video.setdefault(fp.name.rsplit("__", 1)[0], []).append(fp)
        vids = sorted(by_video)
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_frac)) if len(vids) > 1 else 0
        for v in vids[:n_val]:
            val_items += [(p, idx) for p in by_video[v]]
        for v in vids[n_val:]:
            train_items += [(p, idx) for p in by_video[v]]
    return train_items, val_items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=r"C:\dvad\data\ahc",
                   help="Unused here (the cache is prebuilt); kept so every "
                        "script in src/ takes the same flag.")
    p.add_argument("--cache", default=r"C:\dvad\data\motion_frames")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Keep every epoch, not just best-by-val: val recall and "
                        "test macro-F1 have already diverged once on this project.")
    p.add_argument("--out", default=str(Path(MODELS_DIR) / "motion_classifier.pt"))
    args = p.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(f"No motion cache at {cache} - run src\\build_motion_frames.py first")

    train_items, val_items = build_splits(cache, args.val_frac, args.seed)
    if not train_items:
        raise SystemExit(f"No frames found under {cache}")
    counts = [sum(1 for _, l in train_items if l == i) for i in range(len(CLASSES))]
    print(f"[data] train={len(train_items)} val={len(val_items)} (split by source video)")
    for i, c in enumerate(CLASSES):
        nv = sum(1 for _, l in val_items if l == i)
        print(f"  {c:<34} train={counts[i]:>5}  val={nv:>4}")

    device = args.device if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(FrameDS(train_items, TRAIN_TF), batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(FrameDS(val_items, VAL_TF), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model = model.to(device)

    # SQRT-inverse frequency, clamped at 4x - the same recipe as the appearance
    # trainer, for the same measured reason: plain inverse frequency once handed
    # a rare class a ~20x weight and the network answered with it everywhere.
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

        # Validate per CLIP, the way inference aggregates - frame accuracy
        # answers a question nobody asks at runtime. val_dl is shuffle=False,
        # so batches walk val_items in order.
        model.eval()
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
        # Macro recall, not accuracy: `normal` dominates the count, so a model
        # that always answered `normal` would post a flattering accuracy.
        recalls = [cor[i] / tot[i] for i in range(len(CLASSES)) if tot[i]]
        macro = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"epoch {ep}/{args.epochs} loss={tl/max(len(train_dl),1):.4f} "
              f"macro_recall={macro:.3f}")
        for i, c in enumerate(CLASSES):
            if tot[i]:
                print(f"    {c:<34} {cor[i]:>4}/{tot[i]:<4} = {cor[i]/tot[i]:.3f}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_state_dict": model.state_dict(), "classes": CLASSES,
                   "img_size": IMG_SIZE, "val_macro_recall": macro,
                   "input_kind": "ego_compensated_motion_diff",
                   "lags_s": [0.4, 1.5, 4.0]}
        if macro > best:
            best = macro
            torch.save(payload, out)
            print(f"    -> saved {out} (macro_recall={macro:.3f})")
        if args.save_every_epoch:
            torch.save(payload, out.with_name(f"{out.stem}_ep{ep}{out.suffix}"))


if __name__ == "__main__":
    main()
