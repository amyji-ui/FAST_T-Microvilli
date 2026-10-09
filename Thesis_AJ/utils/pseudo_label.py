#!/usr/bin/env python
# coding: utf-8
"""
Run the trained model on currently-unlabeled images and package its
predictions as a Supervisely-importable project, for model-assisted
labeling: correct the predictions in Supervisely (usually much faster than
labeling from scratch) instead of drawing every mask by hand.

This is the reverse of convert_supervisely_annotations.py: instead of
decoding Supervisely bitmaps into a class-id PNG mask, it encodes a
predicted class-id mask into Supervisely bitmap objects (one object per
connected component per class), with nnCreated=true marking them as
model-generated so they're visually distinguishable from hand-drawn objects
in the Supervisely editor.

Workflow:
  1. python pseudo_label.py --config config.yaml --run-dir <trained run> \
         --images-dir "../Images/TIRF_HEK_CD8AA_Out" --out pseudo_labels
     (--images-dir accepts more than one folder, and skips any image already
     present in the training set, so you never re-predict a labeled image.)
  2. In Supervisely: Import -> "Team files" -> upload the pseudo_labels
     folder (it's already in Supervisely's on-disk project format:
     meta.json + <dataset>/img,ann) as a NEW DATASET inside your existing
     project, so convert_supervisely_annotations.py picks it up alongside
     the hand-labeled dataset automatically (it globs every dataset under
     project-dir).
  3. Open each image and correct the pre-drawn objects: delete wrong ones,
     draw missed ones, adjust boundaries. This is a correction pass, not
     labeling from scratch.
  4. Export the corrected project back into Images/Labeled (overwriting or
     alongside the existing dataset), then rerun convert.sh and
     run_train.sh as usual.

Caveats:
  - The model's F-actin predictions are heavily fragmented (Dice ~0.53,
    ~2x too many objects vs manual) -- expect real correction work there,
    not just spot-checks. Lamellipodia (Dice ~0.73) needs less.
  - Self-training risk: correcting predictions you already half-agree with
    is faster, but a systematic model mistake is easy to miss. Look
    especially hard at anything model_confidence-style checks can't catch --
    F-actin gaps/merges, and periodically re-check against genuinely
    from-scratch labels.
  - `classId` values here are placeholders; Supervisely matches classes by
    title when importing a dataset into an existing project.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import shutil
import zlib
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import train_TIRF
from quantification import load_settings, load_model, predict_mask
import torch


def bool_mask_to_bitmap(mask: np.ndarray) -> dict:
    """Encode a full-canvas boolean mask into a Supervisely bitmap dict
    ({"data": base64 str, "origin": [x, y]}), the inverse of
    convert_supervisely_annotations.base64_to_bool_mask. Tightly crops to the
    mask's bounding box first, matching how Supervisely stores bitmaps."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    ok, png_bytes = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Failed to encode mask crop as PNG")
    data = base64.b64encode(zlib.compress(png_bytes.tobytes())).decode("ascii")
    return {"data": data, "origin": [int(x0), int(y0)]}


def load_real_classes(project_dir: str) -> dict:
    """Read {project_dir}/meta.json and return {class title: class entry}.
    Used so pseudo-labels reuse Supervisely's OWN id/color/hotkey for each
    class -- reusing our internal training-mask index (1, 2, ...) as a fake
    "classId" is what caused Supervisely to treat pseudo-labeled classes as
    different from the hand-labeled ones and create "..._1" duplicates,
    even after the shape field was fixed to match."""
    if not project_dir:
        return {}
    meta_path = os.path.join(project_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return {c["title"]: c for c in json.load(f)["classes"]}


def mask_to_objects(mask: np.ndarray, class_name_to_index: dict, real_classes: dict, min_object_area: int) -> list:
    """One Supervisely bitmap object per connected component per foreground
    class (skipping components smaller than min_object_area px), each
    tagged nnCreated=true. classId is the class's REAL Supervisely id from
    real_classes when known, so it's recognized as the same class on import
    (see load_real_classes) -- our internal training-mask index (class_id
    from class_name_to_index) is only used to read the right pixels out of
    `mask`, never written to the output."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    objects = []
    next_id = 1
    for class_name, class_id in class_name_to_index.items():
        real_id = real_classes[class_name]["id"] if class_name in real_classes else class_id
        labeled, n = ndi.label(mask == class_id, structure=np.ones((3, 3)))
        for component in range(1, n + 1):
            component_mask = labeled == component
            if component_mask.sum() < min_object_area:
                continue
            objects.append({
                "id": next_id,
                "classId": real_id,
                "objectId": None,
                "description": "",
                "geometryType": "bitmap",
                "nnCreated": True,
                "nnUpdated": False,
                "labelerLogin": "pseudo_label.py",
                "createdAt": now,
                "updatedAt": now,
                "tags": [],
                "classTitle": class_name,
                "bitmap": bool_mask_to_bitmap(component_mask),
            })
            next_id += 1
    return objects


def build_meta(class_name_to_index: dict, colormap: np.ndarray, real_classes: dict) -> dict:
    classes = []
    for class_name, class_id in class_name_to_index.items():
        if class_name in real_classes:
            # Reuse the real class entry verbatim (id, color, shape, hotkey,
            # description) so Supervisely recognizes it as the SAME class on
            # import, not a new one under the same title.
            classes.append(real_classes[class_name])
            continue
        # Fallback for a class name not found in the project (e.g. testing
        # without --config pointed at the real project, or a genuinely new
        # class): synthesize an entry. This WILL create a new Supervisely
        # class rather than merging -- fine for a new class, a problem if
        # this fires for a class that actually already exists (check
        # project-dir in --config is correct if you see this warning).
        print(f"  ! '{class_name}' not found in the project's meta.json -- writing a new class definition, not reusing an existing one")
        r, g, b = (int(v) for v in colormap[class_id])
        classes.append({
            "title": class_name,
            "description": "",
            "shape": "any",
            "color": f"#{r:02X}{g:02X}{b:02X}",
            "geometry_config": {},
            "id": class_id,
            "hotkey": "",
        })
    return {"classes": classes, "tags": [], "projectType": "images", "projectSettings": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder (model/best_model.pth). Default: config's out-dir.")
    parser.add_argument("--images-dir", nargs="+", required=True, help="One or more folders of preprocessed (grayscale PNG) unlabeled images.")
    parser.add_argument("--out", required=True, help="Output folder, in Supervisely's on-disk project format.")
    parser.add_argument("--dataset-name", default="pseudo_labeled", help="Dataset folder name inside --out.")
    parser.add_argument("--min-object-area", type=int, default=4, help="Skip predicted objects smaller than this (px^2); avoids flooding the editor with single-pixel noise.")
    parser.add_argument("--limit", type=int, default=None, help="Only process N new images, randomly sampled (e.g. for a quick trial batch).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --limit's sampling, so the same batch is reproducible.")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    # Dedupe by pixel content, not filename: Supervisely renames images on
    # upload, so an already-labeled image's filename in --images-dir usually
    # won't match its (different) name in the training set.
    already_labeled_hashes = {
        hashlib.md5(np.array(Image.open(train_TIRF.resolve_image_path(settings["image_dir"], n)).convert("L")).tobytes()).hexdigest()
        for n in train_TIRF.list_dataset_images(settings["image_dir"])
    }

    candidates, skipped = [], 0
    for folder in args.images_dir:
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".png"):
                continue
            path = os.path.join(folder, name)
            content_hash = hashlib.md5(np.array(Image.open(path).convert("L")).tobytes()).hexdigest()
            if content_hash in already_labeled_hashes:
                skipped += 1
                continue
            candidates.append((folder, name))
    if skipped:
        print(f"Skipping {skipped} image(s) already in the training set (matched by pixel content).")
    if not candidates:
        raise SystemExit("No new images found (every image in --images-dir is already in the training set).")
    if args.limit:
        candidates = random.Random(args.seed).sample(candidates, min(args.limit, len(candidates)))

    img_dir = os.path.join(args.out, args.dataset_name, "img")
    ann_dir = os.path.join(args.out, args.dataset_name, "ann")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    class_name_to_index = {name: cid for cid, name in settings["class_names"].items() if cid != 0}
    real_classes = load_real_classes(settings["project_dir"])
    missing = [n for n in class_name_to_index if n not in real_classes]
    if missing:
        print(f"WARNING: {missing} not found in {settings['project_dir']}/meta.json -- check --config's project-dir is your real labeled project, or these will import as new classes.")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(build_meta(class_name_to_index, settings["colormap"], real_classes), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(settings["run_dir"], settings["classes"], device)
    print(f"Predicting {len(candidates)} new images from {[os.path.basename(f) for f in args.images_dir]} on {device}")

    n_objects = 0
    for folder, name in candidates:
        src_path = os.path.join(folder, name)
        pred = predict_mask(model, src_path, device)
        with Image.open(src_path) as im:
            height, width = im.height, im.width
        objects = mask_to_objects(pred, class_name_to_index, real_classes, args.min_object_area)
        n_objects += len(objects)
        shutil.copy2(src_path, os.path.join(img_dir, name))
        with open(os.path.join(ann_dir, name + ".json"), "w") as f:
            json.dump({"description": "", "tags": [], "size": {"height": height, "width": width}, "objects": objects}, f, indent=4)

    print(f"Wrote {len(candidates)} images and {n_objects} pre-drawn objects ({n_objects / len(candidates):.0f}/image) to {args.out}")
    print(f"Import {args.out} into Supervisely as a new dataset, correct the objects, export back into {settings.get('project_dir', 'your Labeled project')}, then rerun convert.sh + run_train.sh.")


if __name__ == "__main__":
    main()
