"""Training loop: random crops, BCE + Dice, full-image validation scored with official PQ."""
from __future__ import annotations

import copy
import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import CropDataset, read_image
from .infer import predict_image
from .metrics import evaluate
from .model import bce_dice_loss, bce_lovasz_loss, build_model, save_checkpoint
from .postprocess import PostCfg, instances
from .submission import to_dataframe


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML config + dotted overrides like ["train.epochs=2", "fold=1"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def predict_stems(model, stems, img_dir, icfg, device) -> dict[str, np.ndarray]:
    probs = {}
    for stem in stems:
        img = read_image(Path(img_dir) / f"{stem}.png")
        probs[stem] = predict_image(model, img, tile=icfg["tile"], overlap=icfg["overlap"],
                                    tta=icfg.get("tta", False), batch=icfg["batch"], device=device)
    return probs


def score_probs(probs: dict, gt_df: pd.DataFrame, post: PostCfg):
    preds = {s: [c["counts"] for c in instances(p, post)] for s, p in probs.items()}
    gt = gt_df[gt_df["stem"].isin(list(probs))]
    return evaluate(gt[["filament_id", "segmentation_rle"]], to_dataframe(preds))


def run(cfg: dict) -> None:
    seed_everything(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    if (work / "done").exists() or (work / "val_summary.yaml").exists():
        print(f"{work} already finished, skipping (delete the folder to retrain)")
        return
    resume_path = work / "last.pt"
    resuming = resume_path.exists() and cfg.get("resume", True)
    if not resuming:
        with open(work / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    cache = Path(cfg["cache_dir"])
    folds = pd.read_csv(cache / "folds.csv")
    gt_df = pd.read_csv(cache / "gt.csv")
    fold = cfg.get("fold", 0)
    if fold >= 0:
        tr_stems = folds.loc[folds["fold"] != fold, "stem"].tolist()
        va_stems = folds.loc[folds["fold"] == fold, "stem"].tolist()
    else:  # full-data refit, no validation
        tr_stems, va_stems = folds["stem"].tolist(), []
    img_dir = cache / f"img_{cfg['preprocess']}"
    lab_dir = cache / f"label_{cfg['label_mode']}"
    print(f"device {device} | train {len(tr_stems)} images | val {len(va_stems)} images")

    t = cfg["train"]
    ds = CropDataset(tr_stems, img_dir, lab_dir, crop=t["crop"], samples_per_epoch=t["samples_per_epoch"],
                     pos_frac=t["pos_frac"], augment=True)
    dl = DataLoader(ds, batch_size=t["batch_size"], num_workers=t["num_workers"], drop_last=True,
                    pin_memory=device == "cuda", worker_init_fn=_worker_init,
                    persistent_workers=t["num_workers"] > 0)

    m = cfg["model"]
    model = build_model(m["arch"], m["encoder"], m.get("weights", "imagenet"), m.get("in_channels", 1)).to(device)
    if m.get("init_from") and not resuming:
        # fine-tune from an already trained model (e.g. switch the loss for the last epochs)
        ck0 = torch.load(m["init_from"], map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck0["state_dict"].items()})
        print(f"initialised from {m['init_from']}")
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    total = t["epochs"] * len(dl)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = t.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    post = PostCfg(**cfg["post"])
    icfg_val = {**cfg["infer"], "tta": False}

    log_path = work / "log.csv"
    best, start_ep = -1.0, 1
    if resuming:
        # Continue an interrupted run: model, optimizer, LR schedule and AMP scaler state.
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if "opt" not in ck:
            raise SystemExit(f"{resume_path} has no optimizer state (old format); delete {work} to restart")
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, start_ep = ck.get("best", -1.0), ck["epoch"] + 1
        print(f"resuming from epoch {start_ep} (best val PQ so far {best:.4f})")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_pq", "val_sq", "val_rq", "tp", "fp", "fn", "lr", "sec"])
    for ep in range(start_ep, t["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
            if t.get("loss", "bce_dice") == "bce_lovasz":
                loss = bce_lovasz_loss(logits.float(), y, t.get("lovasz_weight", 1.0))
            else:
                loss = bce_dice_loss(logits.float(), y, t.get("dice_weight", 1.0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        row = [ep, float(np.mean(losses)), "", "", "", "", "", "", opt.param_groups[0]["lr"], 0]
        if va_stems and (ep % t["val_every"] == 0 or ep == t["epochs"]):
            model.eval()
            res = score_probs(predict_stems(model, va_stems, img_dir, icfg_val, device), gt_df, post)
            row[2:8] = [res.pq, res.sq, res.rq, res.tp, res.fp, res.fn]
            if res.pq > best:
                best = res.pq
                save_checkpoint(work / "best.pt", model, cfg, epoch=ep, val_pq=res.pq)
            print(f"ep {ep:3d} loss {row[1]:.4f} | {res.summary()}")
        else:
            print(f"ep {ep:3d} loss {row[1]:.4f}")
        row[-1] = round(time.time() - t0, 1)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        # Written to a temp file first so an interruption never leaves a corrupt last.pt.
        save_checkpoint(work / "last.tmp", model, cfg, epoch=ep, best=best, opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict())
        (work / "last.tmp").replace(work / "last.pt")

    # Save validation probabilities of the best model for post-processing tuning.
    if va_stems:
        from .model import load_checkpoint
        best_model, _ = load_checkpoint(work / "best.pt", device)
        icfg = copy.deepcopy(cfg["infer"])
        probs = predict_stems(best_model, va_stems, img_dir, icfg, device)
        pdir = work / "val_probs"
        pdir.mkdir(exist_ok=True)
        for s, p in probs.items():
            np.save(pdir / f"{s}.npy", p.astype(np.float16))
        res = score_probs(probs, gt_df, post)
        print(f"best model, TTA={icfg.get('tta')}: {res.summary()}")
        with open(work / "val_summary.yaml", "w") as f:
            yaml.safe_dump({"pq": res.pq, "sq": res.sq, "rq": res.rq, "tp": res.tp, "fp": res.fp,
                            "fn": res.fn, "post": post.to_dict(), "tta": icfg.get("tta")}, f)
    (work / "done").touch()
