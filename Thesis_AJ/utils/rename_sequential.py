#!/usr/bin/env python
"""
Rename every image in a folder to <prefix>_<index>.<ext>, sequentially, and
write rename_manifest.csv (new name -> original name) alongside them so the
original filenames (and whatever acquisition info is embedded in them)
aren't lost.

Safe to re-run after adding/deleting images: if rename_manifest.csv already
exists, it's used to look up each file's TRUE original name (chaining through
however many renames it's already been through) before overwriting it, so
the provenance link survives even though the sequential numbers shift. A
file deleted since the last run simply drops out of the new manifest instead
of leaving a stale, mismatched row.

Usage:
    python rename_sequential.py <folder> <prefix>
    python rename_sequential.py ../Images/Source/TIRF_HEK_CD8BB_Out CD8BB
    python rename_sequential.py <folder> <prefix> --start 1 --digits 3
"""

import argparse
import csv
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder of images to rename, in place.")
    parser.add_argument("prefix", help="New filename prefix, e.g. CD8AA -> CD8AA_00.png, CD8AA_01.png, ...")
    parser.add_argument("--ext", default="png", help="Image extension to match (default: png).")
    parser.add_argument("--start", type=int, default=0, help="First index (default: 0).")
    parser.add_argument("--digits", type=int, default=None,
                         help="Zero-padded digit count. Default: enough to fit the last index, minimum 2 (so '00' not '0').")
    args = parser.parse_args()

    ext = args.ext.lstrip(".").lower()
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(f".{ext}"))
    n = len(files)
    print(f"{n} .{ext} image(s) in {args.folder}")
    if n == 0:
        return

    manifest_path = os.path.join(args.folder, "rename_manifest.csv")
    true_original = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            true_original = {row["new_name"]: row["original_name"] for row in csv.DictReader(f)}
        reused = sum(1 for name in files if name in true_original)
        print(f"Found existing manifest: resolved {reused}/{n} current filenames to their true original name.")

    digits = args.digits or max(2, len(str(args.start + n - 1)))
    new_names = [f"{args.prefix}_{args.start + i:0{digits}d}.{ext}" for i in range(n)]
    # Chain through the existing manifest so re-running after a delete/add
    # still traces back to the ORIGINAL raw filename, not just the name a
    # file happened to have on disk before this run.
    rows = [(new, true_original.get(old, old)) for new, old in zip(new_names, files)]

    # Two-pass rename (old -> temp -> new) so it's safe even if a new name
    # happens to collide with a not-yet-renamed old name. Renames the actual
    # on-disk names (`files`), not the manifest-resolved ones in `rows`.
    temp_names = [f".tmp_{i}.{ext}" for i in range(n)]
    for old, temp in zip(files, temp_names):
        os.rename(os.path.join(args.folder, old), os.path.join(args.folder, temp))
    for new_name, temp in zip(new_names, temp_names):
        os.rename(os.path.join(args.folder, temp), os.path.join(args.folder, new_name))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "original_name"])
        writer.writerows(rows)

    print(f"Renamed {n} image(s) -> {rows[0][0]} .. {rows[-1][0]}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
