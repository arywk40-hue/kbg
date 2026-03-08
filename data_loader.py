"""
data_loader.py – Dataset loading, augmentation, and DataLoader creation.

Supports:
  • Multi-class (or binary) fracture classification from ImageFolder-style layout
  • Albumentations-based augmentation pipeline (train / val-test)
  • MixUp collate function
  • 5-fold stratified cross-validation splits
  • Class-balanced sampling for imbalanced datasets
"""

import os
import random
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Generator

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transforms(cfg: dict) -> A.Compose:
    """Build training augmentation pipeline using Albumentations."""
    aug = cfg["augmentation"]
    norm = aug["normalize"]
    img_size = cfg["data"]["img_size"]

    transforms = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5 if aug["train"]["random_horizontal_flip"] else 0.0),
        A.Rotate(limit=aug["train"]["random_rotation"], p=0.5),
        A.ColorJitter(
            brightness=aug["train"]["color_jitter"]["brightness"],
            contrast=aug["train"]["color_jitter"]["contrast"],
            p=0.4,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=aug["train"]["gaussian_blur_prob"]),
        A.Affine(
            translate_percent={"x": (-aug["train"]["random_affine"]["translate"][0],
                                      aug["train"]["random_affine"]["translate"][0]),
                               "y": (-aug["train"]["random_affine"]["translate"][1],
                                      aug["train"]["random_affine"]["translate"][1])},
            scale=(aug["train"]["random_affine"]["scale"][0],
                   aug["train"]["random_affine"]["scale"][1]),
            rotate=(-aug["train"]["random_affine"]["degrees"],
                     aug["train"]["random_affine"]["degrees"]),
            p=0.5,
        ),
        A.GridDistortion(p=0.15),           # simulate X-ray distortions
        A.CLAHE(clip_limit=2.0, p=0.3),     # enhance local contrast (X-ray specific)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 32, img_size // 16),
            hole_width_range=(img_size // 32, img_size // 16),
            p=aug["train"]["random_erasing_prob"],
        ),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def build_val_transforms(cfg: dict) -> A.Compose:
    """Build validation / test transform pipeline (resize + normalize only)."""
    norm = cfg["augmentation"]["normalize"]
    img_size = cfg["data"]["img_size"]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FractureDataset(Dataset):
    """
    Reads images from an ImageFolder-style directory tree.

        data_dir/
          class_A/img1.png
          class_B/img2.png
          ...

    Parameters
    ----------
    root       : path to split directory (train / val / test)
    transform  : albumentations Compose pipeline
    """

    def __init__(self, root: str, transform: Optional[A.Compose] = None):
        self.root = Path(root)
        self.transform = transform

        self.classes: List[str] = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Try cv2 first, fall back to PIL
        image = cv2.imread(str(img_path))
        if image is None:
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                # Return a blank image instead of crashing
                img_size = 224
                image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_labels(self) -> List[int]:
        return self.targets


# ─────────────────────────────────────────────────────────────────────────────
# TransformSubset – wraps a Subset with its own transform
# ─────────────────────────────────────────────────────────────────────────────

class TransformSubset(Dataset):
    """
    A Subset that applies its own transform, avoiding the pitfall of mutating
    the underlying dataset's transform (which would affect all folds).
    """

    def __init__(self, dataset: FractureDataset, indices: np.ndarray,
                 transform: Optional[A.Compose] = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]

        image = cv2.imread(str(img_path))
        if image is None:
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_labels(self) -> List[int]:
        return self.targets


# ─────────────────────────────────────────────────────────────────────────────
# Class-balanced sampler
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset) -> torch.utils.data.WeightedRandomSampler:
    """Create a weighted random sampler to handle class imbalance."""
    if hasattr(dataset, "get_labels"):
        labels = np.array(dataset.get_labels())
    elif hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# MixUp collate
# ─────────────────────────────────────────────────────────────────────────────

class MixUpCollate:
    """Picklable collate_fn that applies MixUp with probability 0.5."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        if self.alpha > 0 and random.random() < 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            perm = torch.randperm(images.size(0))
            images = lam * images + (1 - lam) * images[perm]
            labels_a, labels_b = labels, labels[perm]
            return images, (labels_a, labels_b, lam)
        return images, labels


def mixup_collate(alpha: float = 0.2):
    """Return a picklable MixUpCollate instance."""
    return MixUpCollate(alpha)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> dict:
    """
    Build train / val / test DataLoaders from config.

    Returns
    -------
    dict with keys: 'train', 'val', 'test', 'classes', 'num_classes',
                    'train_ds', 'val_ds', 'test_ds'
    """
    seed_everything(cfg["training"]["seed"])

    train_ds = FractureDataset(cfg["data"]["train_dir"], build_train_transforms(cfg))
    val_ds   = FractureDataset(cfg["data"]["val_dir"],   build_val_transforms(cfg))
    test_ds  = FractureDataset(cfg["data"]["test_dir"],  build_val_transforms(cfg))

    sampler = make_weighted_sampler(train_ds)

    # Determine num_workers and pin_memory based on platform
    num_workers = cfg["data"]["num_workers"]
    pin_memory  = cfg["data"]["pin_memory"]
    import platform
    if platform.system() == "Darwin":
        # macOS: multiprocessing can cause pickling issues; disable pin_memory on MPS
        num_workers = 0
        pin_memory  = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mixup_collate(cfg["augmentation"]["train"]["mixup_alpha"]),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Data] Classes: {train_ds.classes}")
    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return {
        "train":       train_loader,
        "val":         val_loader,
        "test":        test_loader,
        "classes":     train_ds.classes,
        "num_classes": len(train_ds.classes),
        "train_ds":    train_ds,
        "val_ds":      val_ds,
        "test_ds":     test_ds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-fold stratified CV splits
# ─────────────────────────────────────────────────────────────────────────────

def build_cv_loaders(cfg: dict, full_train_dir: Optional[str] = None) -> Generator:
    """
    Yield (fold_idx, train_loader, val_loader) for 5-fold stratified CV.
    Uses the training split only.
    """
    full_train_dir = full_train_dir or cfg["data"]["train_dir"]
    full_ds = FractureDataset(full_train_dir, transform=None)  # no transform; TransformSubset handles it
    labels  = np.array(full_ds.get_labels())

    skf = StratifiedKFold(
        n_splits=cfg["training"]["cross_validation"]["folds"],
        shuffle=True,
        random_state=cfg["training"]["seed"],
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Use TransformSubset so each fold gets its own transform safely
        train_subset = TransformSubset(full_ds, train_idx, build_train_transforms(cfg))
        val_subset   = TransformSubset(full_ds, val_idx,   build_val_transforms(cfg))

        # Determine num_workers and pin_memory based on platform
        num_workers = cfg["data"]["num_workers"]
        pin_memory  = cfg["data"].get("pin_memory", True)
        import platform
        if platform.system() == "Darwin":
            num_workers = 0
            pin_memory  = False

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        yield fold, train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    loaders = build_dataloaders(cfg)
    print(f"Classes     : {loaders['classes']}")
    print(f"Num classes : {loaders['num_classes']}")
    images, labels = next(iter(loaders["train"]))
    if isinstance(labels, tuple):
        labels_a, labels_b, lam = labels
        print(f"Train batch (MixUp) : images={images.shape}, lam={lam:.3f}")
    else:
        print(f"Train batch : images={images.shape}, labels={labels.shape}")
