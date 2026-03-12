"""
train.py – Full training loop for bone fracture classification.

Features
────────
  • Mixed precision (fp16) training via torch.amp
  • Gradient accumulation for effective larger batch sizes
  • Cosine LR schedule with linear warmup
  • Early stopping on validation macro-F1
  • Per-epoch CSV logging → model_performance_analysis.csv
  • 5-fold stratified cross-validation option
  • Best checkpoint auto-saving
  • Apple Silicon (MPS) and CUDA support
"""

import os
import time
import yaml
import csv
import math
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, accuracy_score

from data_loader import build_dataloaders, build_cv_loaders, seed_everything, load_config
from model import build_model, build_criterion, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Device selection (CUDA > MPS > CPU)
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler: cosine with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["scheduler"]["warmup_epochs"]
    min_lr_ratio  = cfg["training"]["scheduler"]["min_lr"] / cfg["training"]["optimizer"]["lr"]

    total_steps  = total_epochs  * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler: Optional[GradScaler],
    scheduler,
    device: torch.device,
    accumulation_steps: int,
    clip_grad: float,
) -> Tuple[float, float]:

    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    optimizer.zero_grad()
    use_amp = scaler is not None

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # Handle MixUp labels
        is_mixup = isinstance(labels, (tuple, list)) and len(labels) == 3
        if is_mixup:
            labels_a, labels_b, lam = labels
            labels_a = labels_a.to(device)
            labels_b = labels_b.to(device)
            mixed_labels = (labels_a, labels_b, lam)
        else:
            if isinstance(labels, (list, tuple)):
                labels = torch.as_tensor(labels, device=device)
            else:
                labels = labels.to(device)

        # Only CUDA supports torch.amp autocast; fallback to no-op elsewhere.
        amp_ctx = autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            logits = model(images)
            if is_mixup:
                loss = criterion(logits, mixed_labels)
                hard_labels = labels_a  # for accuracy tracking
            else:
                loss = criterion(logits, labels)
                hard_labels = labels
            loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(hard_labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Validation / evaluation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float, float]:

    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger for model_performance_analysis.csv
# ─────────────────────────────────────────────────────────────────────────────

class EpochLogger:
    HEADER = [
        "epoch", "train_loss", "val_loss", "train_accuracy",
        "val_accuracy", "val_macro_f1", "overfitting_gap", "learning_rate",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc, val_f1, lr):
        gap = val_loss - train_loss
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                f"{train_acc:.5f}", f"{val_acc:.5f}", f"{val_f1:.5f}",
                f"{gap:.5f}", f"{lr:.8f}",
            ])

    def append_summary(self, summary: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["GENERALIZATION METRICS"])
            for k, v in summary.items():
                writer.writerow([k, v])


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience  = patience
        self.mode      = mode
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        improved = (self.mode == "max" and value > self.best) or \
                   (self.mode == "min" and value < self.best)
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return improved


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, fold: Optional[int] = None, loaders: Optional[dict] = None):
    seed_everything(cfg["training"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    if loaders is None:
        loaders = build_dataloaders(cfg)

    num_classes = loaders["num_classes"]
    model       = build_model(cfg, num_classes).to(device)
    criterion   = build_criterion(cfg).to(device)

    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["optimizer"]["lr"],
        weight_decay=cfg["training"]["optimizer"]["weight_decay"],
        betas=tuple(cfg["training"]["optimizer"]["betas"]),
    )

    # AMP scaler only for CUDA (MPS does not support GradScaler)
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    # Output paths
    out_dir  = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix   = f"_fold{fold}" if fold is not None else ""
    ckpt_path = out_dir / f"best_model{suffix}.pth"

    log_path = cfg["output"]["performance_csv"].replace(".csv", f"{suffix}.csv")
    logger   = EpochLogger(log_path)
    stopper  = EarlyStopping(cfg["training"]["early_stopping"]["patience"], mode="max")

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t_epoch = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, scaler, scheduler,
            device, cfg["training"]["accumulation_steps"], cfg["training"]["gradient_clip"],
        )
        val_loss, val_acc, val_f1 = evaluate(model, loaders["val"], criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.log(epoch, train_loss, val_loss, train_acc, val_acc, val_f1, lr)

        elapsed = time.time() - t_epoch
        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"TL={train_loss:.4f} TA={train_acc:.4f} | "
            f"VL={val_loss:.4f} VA={val_acc:.4f} VF1={val_f1:.4f} | "
            f"LR={lr:.6f} | {elapsed:.1f}s"
        )

        improved = stopper.step(val_f1)
        if improved:
            best_epoch = epoch
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_f1":       val_f1,
                "val_accuracy": val_acc,
                "classes":      loaders["classes"],
            }, ckpt_path)
            print(f"  >>> Checkpoint saved (val_f1={val_f1:.4f})")

        if stopper.triggered:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} min. Best epoch: {best_epoch}")

    # Append summary to CSV
    best_val_acc = max(history["val_acc"])
    max_gap      = max(v - t for v, t in zip(history["val_loss"], history["train_loss"]))
    logger.append_summary({
        "Max_Overfitting_Gap":  f"{max_gap*100:.2f}%",
        "Best_Val_Accuracy":    f"{best_val_acc*100:.2f}% (epoch {best_epoch})",
        "Best_Val_F1":          f"{max(history['val_f1'])*100:.2f}%",
        "Total_Training_Time":  f"{total_time/60:.1f} min",
    })

    return ckpt_path, history


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(cfg: dict):
    """Run 5-fold CV and report mean +/- std validation F1."""
    fold_f1s = []
    for fold, train_loader, val_loader in build_cv_loaders(cfg):
        print(f"\n{'='*50}\n  FOLD {fold + 1} / {cfg['training']['cross_validation']['folds']}\n{'='*50}")
        # Build a full loader set (for num_classes, classes info)
        loaders = build_dataloaders(cfg)
        loaders["train"] = train_loader
        loaders["val"]   = val_loader
        _, history = train(cfg, fold=fold + 1, loaders=loaders)
        fold_f1s.append(max(history["val_f1"]))
        print(f"  Fold {fold+1} best val F1: {fold_f1s[-1]:.4f}")

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    print(f"\n5-Fold CV: {mean_f1*100:.2f}% +/- {std_f1*100:.2f}% (F1)")
    return mean_f1, std_f1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bone fracture classifier")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.cv:
        run_cross_validation(cfg)
    else:
        train(cfg)
