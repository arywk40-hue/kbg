"""
dedupe_report.py — detect duplicate images across train/val/test splits.

Finds both exact duplicates (SHA1) and near-duplicates (perceptual hash),
reports how many cross-split collisions exist, and optionally writes a CSV
with all offending files.
"""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import imagehash


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img.convert("RGB")))


def iter_images(root: Path, splits: List[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.suffix.lower() in IMG_EXTS and path.is_file():
                yield split, path


def build_hash_table(
    root: Path, splits: List[str], hash_fn, hash_name: str
) -> Dict[str, List[Tuple[str, Path]]]:
    table: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for split, path in iter_images(root, splits):
        try:
            h = hash_fn(path)
            table[h].append((split, path))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Skipping {path} ({hash_name} failed: {exc})")
    return table


def find_overlaps(table: Dict[str, List[Tuple[str, Path]]]) -> Dict[str, List[Tuple[str, Path]]]:
    overlaps = {}
    for h, items in table.items():
        splits = {s for s, _ in items}
        if len(splits) > 1:
            overlaps[h] = items
    return overlaps


def write_csv(path: Path, overlaps: Dict[str, List[Tuple[str, Path]]], hash_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hash_type", "hash", "split", "path"])
        for h, items in overlaps.items():
            for split, p in items:
                writer.writerow([hash_name, h, split, str(p)])


def main():
    parser = argparse.ArgumentParser(description="Detect duplicate images across dataset splits.")
    parser.add_argument("--root", default="data/Bone_Fracture_Binary_Classification",
                        help="Dataset root containing train/val/test.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to scan.")
    parser.add_argument("--output", default=None,
                        help="CSV path to save duplicate listings (optional).")
    parser.add_argument("--skip-phash", action="store_true",
                        help="Skip perceptual hash (faster, but misses near-duplicates).")
    parser.add_argument("--preview", type=int, default=10,
                        help="How many overlap groups to print per hash type.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    print(f"Scanning exact duplicates (SHA1) under {root} ...")
    sha_table = build_hash_table(root, args.splits, sha1, "sha1")
    sha_overlaps = find_overlaps(sha_table)
    print(f"Found {len(sha_overlaps)} cross-split exact-duplicate groups.")

    for h, items in list(sha_overlaps.items())[: args.preview]:
        print(f"  {h[:12]}…")
        for split, p in items:
            print(f"    [{split}] {p}")

    ph_overlaps = {}
    if not args.skip_phash:
        print("\nScanning perceptual duplicates (pHash) ...")
        ph_table = build_hash_table(root, args.splits, phash, "phash")
        ph_overlaps = find_overlaps(ph_table)
        print(f"Found {len(ph_overlaps)} cross-split perceptual-duplicate groups.")
        for h, items in list(ph_overlaps.items())[: args.preview]:
            print(f"  {h[:12]}…")
            for split, p in items:
                print(f"    [{split}] {p}")

    if args.output:
        out_path = Path(args.output)
        write_csv(out_path, sha_overlaps, "sha1")
        if ph_overlaps:
            write_csv(out_path.with_suffix(out_path.suffix or ".csv"), ph_overlaps, "phash")
        print(f"\nDuplicate listing saved to {out_path}")

    if sha_overlaps or ph_overlaps:
        print("\n⚠️  Leakage detected across splits. Consider re-splitting after deduplication.")
        raise SystemExit(1)
    print("\n✅ No cross-split duplicates detected.")


if __name__ == "__main__":
    main()
