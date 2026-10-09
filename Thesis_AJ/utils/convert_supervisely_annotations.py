#!/usr/bin/env python
# coding: utf-8
"""
Convert a Supervisely-exported labeling project into the image/mask PNG pairs
that FAST/train.py's ConfocalDataset expects.

Supervisely stores each object as a cropped, base64+zlib+PNG-encoded bitmap
("smart tool" / brush annotations) plus an (x, y) "origin" telling you where
to paste it back onto the full-size image. This script:

  1. Reads <project>/meta.json to get the list of labeled classes.
  2. For every image in <project>/<dataset>/img, loads the matching
     <project>/<dataset>/ann/<image>.json annotation.
  3. Decodes each object's bitmap and paints it onto a blank (H, W) canvas
     using an integer class id (background = 0), at the object's origin.
  4. Saves the canvas as a single-channel "L" mode PNG mask (pixel value ==
     class id, NOT a color) alongside a copy of the source image.
  5. Also saves a colorized image+mask overlay to out/preview/, since the
     raw masks have pixel values 0-3 and look solid black in a normal image
     viewer - use the preview to actually check the labels/paint order.

Usage (reads project-dir/converted-dir/etc. from config.yaml):
    python convert_supervisely_annotations.py --config config.yaml

All parameters can also be passed/overridden on the command line, e.g.:
    python convert_supervisely_annotations.py \
        --config config.yaml --project "../Images/Labeled" --out "../Images/Labeled_converted"

Output mirrors project-dir's own dataset-folder structure: each Supervisely
dataset (e.g. <project-dir>/Train_0/{img,ann}) gets its own
<converted-dir>/Train_0/{images,masks,preview}, instead of one flat pool --
since datasets never share filenames WITHIN their own img/ folder, no
cross-dataset collision handling is needed; each dataset's output is just
its own namespace. train_TIRF.py/quantification.py read directly from
<converted-dir> (not <converted-dir>/images) and gather images across every
dataset subfolder. Pass --no-preview to skip generating each dataset's
preview/.
"""

import argparse
import base64
import json
import os
import re
import shutil
import zlib
from glob import glob

import cv2
import numpy as np
import yaml
from PIL import Image

# The class setup (class-name-to-index, paint-order, colormap, ignored-classes)
# is defined only in config.yaml -- nothing class-specific is hardcoded here,
# so changing the classes is a config-only edit.


def colormap_dict_to_array(colormap: dict) -> np.ndarray:
    max_idx = max(int(k) for k in colormap)
    arr = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for k, v in colormap.items():
        arr[int(k)] = v
    return arr


def base64_to_bool_mask(data: str) -> np.ndarray:
    """Decode a Supervisely bitmap 'data' string into a boolean numpy mask."""
    compressed = base64.b64decode(data)
    raw_png = zlib.decompress(compressed)
    arr = np.frombuffer(raw_png, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Failed to decode bitmap PNG data")
    if decoded.ndim == 3 and decoded.shape[2] >= 4:
        # Use the alpha channel as the mask.
        return decoded[:, :, 3] > 0
    if decoded.ndim == 2:
        return decoded > 0
    raise ValueError(f"Unexpected decoded bitmap shape: {decoded.shape}")


def resolve_class_name(name: str, known_names: set) -> str:
    """Map a Supervisely class title to one already known to this project
    (class-name-to-index or ignored-classes). Handles the "<name>_<digits>"
    duplicate classes Supervisely can create when importing/merging a
    dataset whose class definition doesn't exactly match an existing
    project class (see pseudo_label.py) -- e.g. "F-actin_1" resolves to
    "F-actin" -- so a fresh duplicate doesn't need a manually-added config
    entry every time, and doesn't need Supervisely's own merge to work
    perfectly. Returns `name` unchanged if nothing matches (caller's
    existing "unknown class" handling then applies)."""
    if name in known_names:
        return name
    stripped = re.sub(r"_\d+$", "", name)
    return stripped if stripped != name and stripped in known_names else name


def build_mask(ann: dict, class_name_to_index: dict, paint_order: list, ignored_classes=()) -> np.ndarray:
    height = ann["size"]["height"]
    width = ann["size"]["width"]
    canvas = np.zeros((height, width), dtype=np.uint8)

    known_names = set(class_name_to_index) | set(ignored_classes)
    objects_by_class = {}
    for obj in ann["objects"]:
        raw_name = obj["classTitle"]
        class_name = resolve_class_name(raw_name, known_names)
        if class_name != raw_name:
            print(f"  (treating '{raw_name}' as '{class_name}')")
        objects_by_class.setdefault(class_name, []).append(obj)

    # Paint in a fixed order so overlaps resolve consistently; any class not
    # listed in paint_order is painted first (i.e. can be overwritten).
    # Priority is keyed by resolved class INDEX, not the literal class name
    # string, so an alias name (e.g. "F-actin_1" mapping to the same index as
    # "F-actin" in class_name_to_index) gets the same paint priority as its
    # canonical class without needing its own separate paint_order entry.
    index_priority = {
        class_name_to_index[name]: position for position, name in enumerate(paint_order) if name in class_name_to_index
    }
    ordered_classes = sorted(
        objects_by_class.keys(),
        key=lambda name: index_priority.get(class_name_to_index.get(name), -1),
    )

    for class_name in ordered_classes:
        if class_name not in class_name_to_index:
            if class_name not in ignored_classes:
                print(f"  ! skipping unknown class '{class_name}' (not in class-name-to-index or ignored-classes)")
            continue
        class_idx = class_name_to_index[class_name]
        for obj in objects_by_class[class_name]:
            if obj.get("geometryType") != "bitmap":
                print(f"  ! skipping object with unsupported geometryType={obj.get('geometryType')}")
                continue
            bitmap = obj["bitmap"]
            mask = base64_to_bool_mask(bitmap["data"])
            ox, oy = bitmap["origin"]  # [x, y]
            h, w = mask.shape
            canvas[oy : oy + h, ox : ox + w][mask] = class_idx

    return canvas


def save_preview(img_rgb: np.ndarray, mask: np.ndarray, out_path: str, colormap: np.ndarray, alpha: float = 0.4) -> None:
    """Save a colorized image+mask overlay so the mask is actually visible."""
    color = colormap[mask]
    overlay = ((1 - alpha) * img_rgb + alpha * color).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml; CLI flags below override its values")
    parser.add_argument("--project", default=None, help="Path to the Supervisely project root (contains meta.json). Overrides config's project-dir.")
    parser.add_argument("--out", default=None, help="Output directory; will contain one <dataset>/{images,masks,preview} subfolder per Supervisely dataset. Overrides config's converted-dir.")
    parser.add_argument("--no-preview", action="store_true", help="Skip generating the colorized overlay preview")
    parser.add_argument("--preview-alpha", type=float, default=None, help="Mask color opacity in the preview overlay (0-1)")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    project_dir = args.project or config.get("project-dir")
    out_dir = args.out or config.get("converted-dir")
    if not project_dir or not out_dir:
        parser.error("project dir and output dir are required: set project-dir/converted-dir in --config, or pass --project/--out")

    class_name_to_index = config.get("class-name-to-index")
    paint_order = config.get("paint-order")
    colormap_dict = config.get("colormap")
    if not class_name_to_index or paint_order is None or not colormap_dict:
        parser.error("class-name-to-index, paint-order and colormap must be set in the --config file")
    ignored_classes = config.get("ignored-classes") or []
    preview_enabled = not args.no_preview and config.get("preview", True)
    preview_alpha = args.preview_alpha if args.preview_alpha is not None else config.get("preview-alpha", 0.4)
    colormap = colormap_dict_to_array(colormap_dict)

    meta_path = os.path.join(project_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    known_classes = {c["title"] for c in meta["classes"]}
    unmapped = known_classes - set(class_name_to_index) - set(ignored_classes)
    if unmapped:
        print(f"Warning: classes in meta.json with no class-name-to-index entry: {unmapped}")

    ann_files = sorted(glob(os.path.join(project_dir, "*", "ann", "*.json")))
    if not ann_files:
        raise SystemExit(f"No annotation files found under {project_dir}/*/ann/*.json")

    converted = 0
    datasets_seen = set()
    for ann_path in ann_files:
        dataset_dir = os.path.dirname(os.path.dirname(ann_path))
        dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
        img_name = os.path.basename(ann_path)[: -len(".json")]
        img_path = os.path.join(dataset_dir, "img", img_name)
        if not os.path.exists(img_path):
            print(f"! image not found for annotation {ann_path}, skipping")
            continue

        with open(ann_path, "r") as f:
            ann = json.load(f)

        print(f"Converting {dataset_name}/{img_name} ...")
        mask = build_mask(ann, class_name_to_index, paint_order, ignored_classes)

        # Sanity check mask/image size agree.
        with Image.open(img_path) as im:
            if im.size != (mask.shape[1], mask.shape[0]):
                print(f"  ! WARNING: image size {im.size} != annotation size {(mask.shape[1], mask.shape[0])}")
            img_rgb = np.array(im.convert("RGB"))

        # Per-dataset output (mirrors project-dir's own dataset folders):
        # <out_dir>/<dataset_name>/{images,masks,preview}. Filenames are only
        # ever compared WITHIN one dataset's own img/ folder (where
        # Supervisely already guarantees uniqueness), so no cross-dataset
        # collision handling is needed here.
        images_out = os.path.join(out_dir, dataset_name, "images")
        masks_out = os.path.join(out_dir, dataset_name, "masks")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)
        Image.fromarray(mask, mode="L").save(os.path.join(masks_out, img_name))
        shutil.copy2(img_path, os.path.join(images_out, img_name))
        if preview_enabled:
            preview_out = os.path.join(out_dir, dataset_name, "preview")
            os.makedirs(preview_out, exist_ok=True)
            save_preview(img_rgb, mask, os.path.join(preview_out, img_name), colormap, alpha=preview_alpha)
        datasets_seen.add(dataset_name)
        converted += 1

    print(f"\nDone. Converted {converted} image/mask pairs into {len(datasets_seen)} dataset folder(s) under {out_dir}:")
    for name in sorted(datasets_seen):
        print(f"  {name}/")


if __name__ == "__main__":
    main()
