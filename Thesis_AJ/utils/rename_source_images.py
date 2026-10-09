#!/usr/bin/env python
"""
One-off cleanup for a raw TIRF source folder (e.g. Images/Source/TIRF_HEK_CD8AA_Out):
finds images that are near-duplicate retakes of the same cell (same field of
view, re-shot seconds apart under different gain/phase settings -- these are
NOT byte-identical, since the settings change the pixel values, so plain
hashing can't find them), moves all but one of each duplicate cluster into a
`Duplicates/` subfolder, then renames the remaining images to a clean
sequential scheme. A CSV manifest records new name -> original name (and so,
indirectly, the original acquisition settings/timestamp embedded in it) for
provenance.

Duplicate detection: a fast downsampled, z-scored correlation screen (robust
to brightness/gain differences) finds candidate pairs, then full-resolution
SSIM confirms them -- this is what the quantification pipeline's reviewer
(see FAST/quantification.py's clDice discussion) verified by eye before this
script was written.

Usage:
    python rename_source_images.py <folder> <prefix>
    python rename_source_images.py ../Images/Source/TIRF_HEK_CD8AA_Out CD8AA
"""

import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

CORR_CANDIDATE_THRESHOLD = 0.80  # downsampled correlation: cheap first pass
SSIM_CONFIRM_THRESHOLD = 0.5     # full-res SSIM: confirms it's really the same cell


def find_duplicate_clusters(folder: str, files: list[str]) -> list[list[str]]:
    """Returns groups of 2+ filenames that are near-duplicate retakes of the
    same cell. Singletons (no duplicate) are not included."""
    small = []
    for f in files:
        im = np.array(Image.open(os.path.join(folder, f)).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float64)
        im = (im - im.mean()) / (im.std() + 1e-8)
        small.append(im.ravel())
    small = np.stack(small)
    norm = small / np.linalg.norm(small, axis=1, keepdims=True)
    corr = norm @ norm.T

    n = len(files)
    full_cache = {}

    def load_full(idx):
        if idx not in full_cache:
            full_cache[idx] = np.array(Image.open(os.path.join(folder, files[idx])).convert("L"), dtype=np.float64)
        return full_cache[idx]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > CORR_CANDIDATE_THRESHOLD:
                if ssim(load_full(i), load_full(j), data_range=255) > SSIM_CONFIRM_THRESHOLD:
                    union(i, j)

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [[files[k] for k in g] for g in groups.values() if len(g) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder of source PNGs to dedupe and rename, in place.")
    parser.add_argument("prefix", help="New filename prefix, e.g. CD8AA -> CD8AA_001.png, CD8AA_002.png, ...")
    parser.add_argument("--digits", type=int, default=3, help="Zero-padded digit count for the sequence number.")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.folder) if f.endswith(".png"))
    print(f"{len(files)} images in {args.folder}")

    clusters = find_duplicate_clusters(args.folder, files)
    dup_dir = os.path.join(args.folder, "Duplicates")
    moved = []
    kept_from_cluster = set()
    for cluster in clusters:
        cluster = sorted(cluster)
        keep, drop = cluster[0], cluster[1:]
        kept_from_cluster.add(keep)
        for name in drop:
            moved.append((name, keep))

    if moved:
        os.makedirs(dup_dir, exist_ok=True)
        for name, keep in moved:
            shutil.move(os.path.join(args.folder, name), os.path.join(dup_dir, name))
            print(f"  duplicate of {keep}: moved {name} -> Duplicates/")

    remaining = sorted(f for f in os.listdir(args.folder) if f.endswith(".png"))
    print(f"{len(remaining)} images remain after moving {len(moved)} duplicate(s) to Duplicates/")

    manifest_path = os.path.join(args.folder, "rename_manifest.csv")
    rows = []
    for i, old_name in enumerate(remaining, start=1):
        new_name = f"{args.prefix}_{i:0{args.digits}d}.png"
        rows.append((new_name, old_name))

    # Two-pass rename (old -> temp -> new) so it's safe even if a new name
    # happens to collide with a not-yet-renamed old name.
    temp_names = [f".tmp_{i}.png" for i in range(len(rows))]
    for (new_name, old_name), temp_name in zip(rows, temp_names):
        os.rename(os.path.join(args.folder, old_name), os.path.join(args.folder, temp_name))
    for (new_name, old_name), temp_name in zip(rows, temp_names):
        os.rename(os.path.join(args.folder, temp_name), os.path.join(args.folder, new_name))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "original_name"])
        writer.writerows(rows)
    print(f"Renamed {len(rows)} images -> {args.prefix}_001.png..{args.prefix}_{len(rows):0{args.digits}d}.png")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
