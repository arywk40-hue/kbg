"""
clean_resplit.py — build leak-free train/val/test splits via stratified sampling.

Workflow:
 1) Gathers all images from existing splits (assumes ImageFolder layout).
 2) Deduplicates by SHA1; warns if the same hash maps to multiple class folders.
 3) Re-splits the unique pool with stratified sampling and writes to a new root.
 4) Uses hardlinks by default to avoid extra disk usage; can copy if preferred.
"""

import argparse
import os
import random
import shutil
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import StratifiedShuffleSplit


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_samples(root: Path, splits: List[str]) -> List[Tuple[Path, str]]:
    samples = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.suffix.lower() in IMG_EXTS and path.is_file():
                cls = path.parent.name
                samples.append((path, cls))
    return samples


def dedupe(samples: List[Tuple[Path, str]]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    unique = []
    seen = {}
    warnings = []
    for path, cls in samples:
        h = sha1(path)
        if h not in seen:
            seen[h] = (path, cls)
            unique.append((path, cls))
        else:
            prev_cls = seen[h][1]
            if prev_cls != cls:
                warnings.append(f"Hash collision with differing labels: {path} ({cls}) vs {seen[h][0]} ({prev_cls})")
    return unique, warnings


def ensure_empty_or_new(dest: Path, force: bool):
    if dest.exists():
        existing = list(dest.rglob("*"))
        if existing and not force:
            raise SystemExit(f"Destination {dest} is not empty. Re-run with --force to overwrite.")
        if force:
            shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)


def place_files(samples: List[Tuple[Path, str]], indices: List[int], dest: Path, split: str, mode: str):
    for idx in indices:
        src, cls = samples[idx]
        out_dir = dest / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / src.name
        if mode == "copy":
            shutil.copy2(src, target)
        else:  # link
            try:
                os.link(src, target)
            except FileExistsError:
                continue
            except OSError:
                shutil.copy2(src, target)


def stratified_indices(labels: List[str], val_ratio: float, test_ratio: float, seed: int):
    X = list(range(len(labels)))
    y = labels
    # First split train vs (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio + test_ratio,
        random_state=seed,
    )
    train_idx, temp_idx = next(sss1.split(X, y))
    temp_labels = [y[i] for i in temp_idx]
    # Split temp into val vs test
    test_size = test_ratio / max(1e-8, (val_ratio + test_ratio))
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    val_rel, test_rel = next(sss2.split(temp_idx, temp_labels))
    val_idx = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Create clean stratified splits without leakage.")
    parser.add_argument("--source", default="data/Bone_Fracture_Binary_Classification",
                        help="Existing dataset root containing train/val/test.")
    parser.add_argument("--dest", default="data/clean_split",
                        help="Output root for new splits.")
    parser.add_argument("--val", type=float, default=0.10, help="Validation ratio (hackathon spec: 0.10).")
    parser.add_argument("--test", type=float, default=0.20, help="Test ratio (hackathon spec: 0.20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mode", choices=["link", "copy"], default="link",
                        help="Hardlink (default) or copy files into dest.")
    parser.add_argument("--force", action="store_true", help="Overwrite dest if it exists.")
    args = parser.parse_args()

    src_root = Path(args.source)
    dest_root = Path(args.dest)
    splits = ["train", "val", "test"]

    if args.val + args.test >= 1.0:
        raise SystemExit("val + test ratios must be < 1.0")

    samples = gather_samples(src_root, splits)
    if not samples:
        raise SystemExit(f"No images found under {src_root}")

    unique, warnings = dedupe(samples)
    if warnings:
        print("\nLabel warnings:")
        for w in warnings:
            print(f"  - {w}")

    labels = [cls for _, cls in unique]
    train_idx, val_idx, test_idx = stratified_indices(labels, args.val, args.test, args.seed)

    ensure_empty_or_new(dest_root, args.force)
    place_files(unique, train_idx, dest_root, "train", args.mode)
    place_files(unique, val_idx,   dest_root, "val",   args.mode)
    place_files(unique, test_idx,  dest_root, "test",  args.mode)

    print(f"\nClean split created at {dest_root}")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val  : {len(val_idx)}")
    print(f"  Test : {len(test_idx)}")
    print(f"Mode: {'hardlinks' if args.mode == 'link' else 'copies'}; Seed: {args.seed}")
    if warnings:
        print("⚠️  Some hashes mapped to multiple class labels. Inspect warnings above.")


if __name__ == "__main__":
    main()
