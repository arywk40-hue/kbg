"""
evaluate.py – Comprehensive evaluation on held-out test set.

Produces
────────
  • results/final_results.csv          – all required metrics
  • results/confusion_matrix.png       – heatmap
  • results/gradcam_samples.png        – GradCAM attention maps
  • results/roc_curves.png             – per-class ROC curves
  • Console summary
"""

import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report,
)
from sklearn.preprocessing import label_binarize

from data_loader import build_dataloaders, load_config
from model import build_model, FractureClassifier, GradCAMWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(cfg: dict, model: torch.nn.Module, device: torch.device) -> dict:
    ckpt_dir  = Path(cfg["output"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from {ckpt_path}  (val_f1={ckpt.get('val_f1', 'N/A')})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (y_true, y_pred, y_proba, inference_time_per_image_ms)
    """
    model.eval()
    all_targets, all_preds, all_probas = [], [], []
    total_images = 0
    t0 = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probas = F.softmax(logits, dim=-1)

        all_targets.extend(labels.numpy())
        all_preds.extend(probas.argmax(dim=-1).cpu().numpy())
        all_probas.append(probas.cpu().numpy())
        total_images += images.size(0)

    elapsed = (time.time() - t0) * 1000  # ms
    inference_time = elapsed / max(1, total_images)

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.vstack(all_probas),
        inference_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: str,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.title("Confusion Matrix - Bone Fracture Classification", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    save_path: str,
) -> float:
    n_cls = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    # For binary case, label_binarize returns a single column
    if n_cls == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(10, 7))
    auc_scores = []
    for i, cls_name in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        cls_auc = auc(fpr, tpr)
        auc_scores.append(cls_auc)
        plt.plot(fpr, tpr, label=f"{cls_name} (AUC={cls_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    macro_auc = np.mean(auc_scores) if auc_scores else 0.0
    plt.title(f"ROC Curves - Macro AUC = {macro_auc:.3f}", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved -> {save_path}")
    return macro_auc


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_samples(
    model: torch.nn.Module,
    dataset,
    classes: List[str],
    device: torch.device,
    save_path: str,
    n_samples: int = 6,
):
    """Plot original X-ray next to GradCAM heatmap for a few test samples."""
    import cv2 as cv2_lib

    # Find the last attention / conv block for GradCAM targeting
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)):
            target_layer = module
    if target_layer is None:
        print("GradCAM: no suitable target layer found, skipping.")
        return

    cam_extractor = GradCAMWrapper(model, target_layer)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        image, label = dataset[idx]
        image_t = image.unsqueeze(0).to(device)

        # GradCAM
        try:
            cam = cam_extractor(image_t)
        except Exception as e:
            print(f"GradCAM failed for sample {idx}: {e}")
            continue

        # Predict
        with torch.no_grad():
            logits = model(image_t)
            pred_idx = logits.argmax(dim=-1).item()

        # De-normalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        cam_np = cam.numpy()
        if cam_np.ndim == 1:       # ViT returns 1D tokens
            side = int(cam_np.shape[0] ** 0.5)
            if side * side < cam_np.shape[0]:
                # Skip CLS token
                cam_np = cam_np[1:]
                side = int(cam_np.shape[0] ** 0.5)
            cam_np = cam_np[:side * side].reshape(side, side)

        cam_resized = cv2_lib.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"GT: {classes[label]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Pred: {classes[pred_idx]}", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("GradCAM", fontsize=9)
    plt.suptitle("GradCAM Attention Maps - Bone Fracture Classifier", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"GradCAM samples saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Build final_results.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_results_csv(
    y_true, y_pred, y_proba, classes,
    macro_auc, inference_time_ms,
    model_size_mb, training_time_min,
    cv_mean, cv_std,
    save_path: str,
):
    n_cls = len(classes)
    per_prec  = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec   = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1    = f1_score(y_true, y_pred, average=None, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r   = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class AUC
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    if n_cls == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_auc = []
    for i in range(n_cls):
        if y_bin[:, i].sum() > 0:
            try:
                per_auc.append(roc_auc_score(y_bin[:, i], y_proba[:, i]))
            except ValueError:
                per_auc.append(float("nan"))
        else:
            per_auc.append(float("nan"))

    rows = []

    def make_row(metric, overall, per_class_vals, interpretation):
        row = {
            "metric_name": metric,
            "overall_value": f"{overall:.4f}",
            **{f"class_{i+1}_value ({classes[i]})": f"{v:.4f}" for i, v in enumerate(per_class_vals)},
            "interpretation": interpretation,
        }
        return row

    rows.append(make_row("Accuracy",   accuracy,  [accuracy] * n_cls, "Overall correctness"))
    rows.append(make_row("Precision",  macro_p,   per_prec,            "Positive prediction reliability"))
    rows.append(make_row("Recall",     macro_r,   per_rec,             "Detection rate per fracture type"))
    rows.append(make_row("F1-Score",   macro_f1,  per_f1,              "Balanced performance per class"))
    rows.append(make_row("AUC-ROC",    macro_auc, per_auc,             "Threshold-independent discriminability"))

    # Scalar metrics
    scalar_rows = [
        {"metric_name": "Macro F1-Score",          "overall_value": f"{macro_f1:.4f}",          "interpretation": "Primary ranking metric"},
        {"metric_name": "AUC-ROC (macro)",          "overall_value": f"{macro_auc:.4f}",          "interpretation": "Secondary ranking metric"},
        {"metric_name": "Inference Time (ms/img)",  "overall_value": f"{inference_time_ms:.2f}",  "interpretation": "Tiebreaker - lower is better"},
        {"metric_name": "Model Size (MB)",          "overall_value": f"{model_size_mb:.1f}",       "interpretation": "Tiebreaker - smaller is better"},
        {"metric_name": "Training Time (min)",      "overall_value": f"{training_time_min:.1f}",   "interpretation": "Informational"},
        {"metric_name": "CV Mean F1 (5-fold)",      "overall_value": f"{cv_mean:.4f}",             "interpretation": "Cross-validation generalization"},
        {"metric_name": "CV Std F1 (5-fold)",       "overall_value": f"{cv_std:.4f}",              "interpretation": "Cross-validation stability"},
    ]

    all_rows = rows + scalar_rows
    df = pd.DataFrame(all_rows)
    cols = ["metric_name", "overall_value"] + \
           [f"class_{i+1}_value ({classes[i]})" for i in range(n_cls)] + \
           ["interpretation"]
    # Fill missing columns
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(save_path, index=False)
    print(f"Results CSV saved -> {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    device = get_device()
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    loaders     = build_dataloaders(cfg)
    num_classes = loaders["num_classes"]
    classes     = loaders["classes"]

    model = build_model(cfg, num_classes).to(device)
    ckpt  = load_checkpoint(cfg, model, device)

    # Inference
    print("\nRunning inference on test set ...")
    y_true, y_pred, y_proba, infer_ms = run_inference(
        model, loaders["test"], device, num_classes
    )
    print(f"Inference time: {infer_ms:.2f} ms/image")

    # Model size
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"
    model_mb  = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0.0

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # Plots
    cm = plot_confusion_matrix(y_true, y_pred, classes,
                               str(results_dir / "confusion_matrix.png"))
    macro_auc = plot_roc_curves(y_true, y_proba, classes,
                                str(results_dir / "roc_curves.png"))
    plot_gradcam_samples(model, loaders["test_ds"], classes, device,
                         str(results_dir / "gradcam_samples.png"))

    # Read CV results from performance CSVs if available
    cv_mean, cv_std = 0.0, 0.0
    results_path = Path(cfg["output"]["results_dir"])
    cv_files = sorted(results_path.glob("model_performance_analysis_fold*.csv"))
    if cv_files:
        fold_f1s = []
        for p in cv_files:
            try:
                df_cv = pd.read_csv(p, on_bad_lines="skip")
                if "val_macro_f1" in df_cv.columns:
                    # Filter only numeric rows
                    numeric = pd.to_numeric(df_cv["val_macro_f1"], errors="coerce")
                    valid = numeric.dropna()
                    if len(valid) > 0:
                        fold_f1s.append(valid.max())
            except Exception:
                continue
        if fold_f1s:
            cv_mean = float(np.mean(fold_f1s))
            cv_std  = float(np.std(fold_f1s))

    # Read training time from performance CSV
    training_time_min = 0.0
    perf_csv = Path(cfg["output"]["performance_csv"])
    if perf_csv.exists():
        try:
            with open(perf_csv, "r") as f:
                content = f.read()
            for line in content.split("\n"):
                if "Total_Training_Time" in line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        training_time_min = float(parts[1].replace("min", "").strip())
        except Exception:
            pass

    # Build results CSV
    build_results_csv(
        y_true, y_pred, y_proba, classes,
        macro_auc=macro_auc,
        inference_time_ms=infer_ms,
        model_size_mb=model_mb,
        training_time_min=training_time_min,
        cv_mean=cv_mean,
        cv_std=cv_std,
        save_path=cfg["output"]["final_results_csv"],
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  Accuracy:      {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"  Macro F1:      {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    print(f"  Macro AUC:     {macro_auc:.4f}")
    print(f"  Inference:     {infer_ms:.2f} ms/image")
    print(f"  Model size:    {model_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
