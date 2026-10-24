#!/usr/bin/env python
"""Convert Supervisely projects into per-dataset image/mask PNG pairs for training."""

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
        return decoded[:, :, 3] > 0
    if decoded.ndim == 2:
        return decoded > 0
    raise ValueError(f"Unexpected decoded bitmap shape: {decoded.shape}")


def resolve_class_name(name: str, known_names: set) -> str:
    """Map a Supervisely class title to one already known to this project (class-name-to-index or ignored-classes)."""
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
            ox, oy = bitmap["origin"]
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

    project_dirs = as_list(args.project or config.get("project-dir"))
    out_dirs = as_list(args.out or config.get("converted-dir"))
    if not project_dirs or not out_dirs:
        parser.error("project dir and output dir are required: set project-dir/converted-dir in --config, or pass --project/--out")
    if len(project_dirs) != len(out_dirs):
        parser.error(f"project-dir lists {len(project_dirs)} folder(s) but converted-dir lists {len(out_dirs)} -- "
                     "they're paired by position (project i -> converted-dir i), so give the same number of each")

    class_name_to_index = config.get("class-name-to-index")
    paint_order = config.get("paint-order")
    colormap_dict = config.get("colormap")
    if not class_name_to_index or paint_order is None or not colormap_dict:
        parser.error("class-name-to-index, paint-order and colormap must be set in the --config file")
    ignored_classes = config.get("ignored-classes") or []
    preview_enabled = not args.no_preview and config.get("preview", True)
    preview_alpha = args.preview_alpha if args.preview_alpha is not None else config.get("preview-alpha", 0.4)
    colormap = colormap_dict_to_array(colormap_dict)

    for project_dir, out_dir in zip(project_dirs, out_dirs):
        print(f"\n=== {project_dir} -> {out_dir} ===")
        convert_project(project_dir, out_dir, class_name_to_index, paint_order, ignored_classes,
                        preview_enabled, preview_alpha, colormap)


def as_list(value) -> list:
    """None -> [], a single path -> [path], a list -> itself."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def convert_project(project_dir, out_dir, class_name_to_index, paint_order, ignored_classes,
                    preview_enabled, preview_alpha, colormap) -> None:
    """Convert one Supervisely project into out_dir/<dataset>/{images,masks,preview}."""
    meta_path = os.path.join(project_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    known_classes = {c["title"] for c in meta["classes"]}
    known_names = set(class_name_to_index) | set(ignored_classes)
    unmapped = {c for c in known_classes if resolve_class_name(c, known_names) not in known_names}
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

        with Image.open(img_path) as im:
            if im.size != (mask.shape[1], mask.shape[0]):
                print(f"  ! WARNING: image size {im.size} != annotation size {(mask.shape[1], mask.shape[0])}")
            img_rgb = np.array(im.convert("RGB"))

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
