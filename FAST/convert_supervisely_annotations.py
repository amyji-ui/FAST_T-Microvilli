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

The resulting <converted-dir>/images and <converted-dir>/masks folders can be
passed directly as image_dir/mask_dir to train_supervisely.py. Pass
--no-preview to skip generating <converted-dir>/preview/.
"""

import argparse
import base64
import json
import os
import shutil
import zlib
from glob import glob

import cv2
import numpy as np
import yaml
from PIL import Image

# Fallback defaults, used for any setting not found in --config and not
# passed on the command line. Must match the class-index convention used by
# train_supervisely.py's --classes 4 (0=background, 1=Actin,
# 2=Focal Adhesions, 3=Lamellipodia). No Filopodia class in this TIRF
# adaptation.
DEFAULT_CLASS_NAME_TO_INDEX = {
    "F-actin": 1,
    "Focal Adhesion": 2,
    "Lamellipodia": 3,
}

# When two labeled objects overlap, the one painted *last* wins. List class
# names here in "paint order" (first = painted first / bottom layer, last =
# painted last / top layer). Adjust to taste for your data.
DEFAULT_PAINT_ORDER = ["Lamellipodia", "F-actin", "Focal Adhesion"]

# RGB colors for the preview overlay, indexed by class id (must line up with
# class-name-to-index / train_supervisely.py's COLORMAP).
DEFAULT_COLORMAP = {
    0: [0, 0, 0],  # Background
    1: [0, 255, 0],  # Actin - Green
    2: [255, 0, 0],  # Focal Adhesions - Red
    3: [255, 255, 0],  # Lamellipodia - Yellow
}


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


def build_mask(ann: dict, class_name_to_index: dict, paint_order: list) -> np.ndarray:
    height = ann["size"]["height"]
    width = ann["size"]["width"]
    canvas = np.zeros((height, width), dtype=np.uint8)

    objects_by_class = {}
    for obj in ann["objects"]:
        objects_by_class.setdefault(obj["classTitle"], []).append(obj)

    # Paint in a fixed order so overlaps resolve consistently; any class not
    # listed in paint_order is painted first (i.e. can be overwritten).
    ordered_classes = sorted(
        objects_by_class.keys(),
        key=lambda name: paint_order.index(name) if name in paint_order else -1,
    )

    for class_name in ordered_classes:
        if class_name not in class_name_to_index:
            print(f"  ! skipping unknown class '{class_name}' (not in CLASS_NAME_TO_INDEX)")
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
    parser.add_argument("--out", default=None, help="Output directory; will contain images/ masks/ preview/ subfolders. Overrides config's converted-dir.")
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

    class_name_to_index = config.get("class-name-to-index", DEFAULT_CLASS_NAME_TO_INDEX)
    paint_order = config.get("paint-order", DEFAULT_PAINT_ORDER)
    preview_enabled = not args.no_preview and config.get("preview", True)
    preview_alpha = args.preview_alpha if args.preview_alpha is not None else config.get("preview-alpha", 0.4)
    colormap = colormap_dict_to_array(config.get("colormap", DEFAULT_COLORMAP))

    images_out = os.path.join(out_dir, "images")
    masks_out = os.path.join(out_dir, "masks")
    preview_out = os.path.join(out_dir, "preview")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)
    if preview_enabled:
        os.makedirs(preview_out, exist_ok=True)

    meta_path = os.path.join(project_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    known_classes = {c["title"] for c in meta["classes"]}
    unmapped = known_classes - set(class_name_to_index)
    if unmapped:
        print(f"Warning: classes in meta.json with no class-name-to-index entry: {unmapped}")

    ann_files = sorted(glob(os.path.join(project_dir, "*", "ann", "*.json")))
    if not ann_files:
        raise SystemExit(f"No annotation files found under {project_dir}/*/ann/*.json")

    converted = 0
    for ann_path in ann_files:
        dataset_dir = os.path.dirname(os.path.dirname(ann_path))
        img_name = os.path.basename(ann_path)[: -len(".json")]
        img_path = os.path.join(dataset_dir, "img", img_name)
        if not os.path.exists(img_path):
            print(f"! image not found for annotation {ann_path}, skipping")
            continue

        with open(ann_path, "r") as f:
            ann = json.load(f)

        print(f"Converting {img_name} ...")
        mask = build_mask(ann, class_name_to_index, paint_order)

        # Sanity check mask/image size agree.
        with Image.open(img_path) as im:
            if im.size != (mask.shape[1], mask.shape[0]):
                print(f"  ! WARNING: image size {im.size} != annotation size {(mask.shape[1], mask.shape[0])}")
            img_rgb = np.array(im.convert("RGB"))

        Image.fromarray(mask, mode="L").save(os.path.join(masks_out, img_name))
        shutil.copy2(img_path, os.path.join(images_out, img_name))
        if preview_enabled:
            save_preview(img_rgb, mask, os.path.join(preview_out, img_name), colormap, alpha=preview_alpha)
        converted += 1

    print(f"\nDone. Converted {converted} image/mask pairs into:")
    print(f"  images: {images_out}")
    print(f"  masks:  {masks_out}")
    if preview_enabled:
        print(f"  preview: {preview_out}")


if __name__ == "__main__":
    main()
