#!/usr/bin/env python
# coding: utf-8
"""
Finds a good per-class confidence threshold (see quantification.py's
predict_mask/--class-threshold) by sweeping a range of values and reporting
Dice/precision/recall at each -- the proper, learned version of the
erosion diagnostic: instead of chopping a fixed number of pixels off every
prediction, this raises the bar a class needs to clear to win a pixel,
letting the model's own confidence decide where to pull back.

Only meaningful on the TEST set (never train/val -- those would just tell
you which threshold overfits the images already trained on).

Usage (from the Thesis_AJ folder):
    python sweep_threshold.py --config config.yaml --run-dir trained_model_01 --class F-actin
    python sweep_threshold.py --config config.yaml --run-dir trained_model_01 --class F-actin --values 0.3,0.4,0.5,0.6,0.7,0.8

Once you've picked a value from the printed table, use it for real with
quantification.py's --class-threshold, e.g.:
    python quantification.py --run-dir trained_model_01 --class-threshold F-actin=0.65
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

import train_TIRF
from quantification import load_settings, load_model, predict_mask, pixel_metrics_bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder. Default: config's out-dir.")
    parser.add_argument("--class", dest="class_name", required=True, help="Class to sweep the threshold for, e.g. F-actin.")
    parser.add_argument("--values", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="Comma-separated threshold values to try.")
    parser.add_argument("--out", default=None, help="CSV output path. Default: <run-dir>/threshold_sweep_<class>.csv")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    run_dir = settings["run_dir"]
    class_names = settings["class_names"]
    name_to_id = {name: cid for cid, name in class_names.items()}
    if args.class_name not in name_to_id:
        raise SystemExit(f"Unknown class '{args.class_name}'; known classes are {list(name_to_id)}")
    class_id = name_to_id[args.class_name]
    values = [float(v) for v in args.values.split(",")]

    current_filenames = train_TIRF.list_dataset_images(settings["image_dir"])
    names = train_TIRF.resolve_pinned_test(current_filenames, settings["image_dir"], settings["test_source_dir"])
    if not names:
        raise SystemExit("No test images found -- check test-source-dir and that they're labeled/converted.")
    print(f"Sweeping '{args.class_name}' threshold over {len(names)} test images from {run_dir}")

    # Also report the merged-classes group (if any) this class belongs to,
    # e.g. microvilli = F-actin + Focal Adhesion -- a threshold change to one
    # member shifts the merged union too, and that's usually the number that
    # actually matters for reporting.
    merged_groups = {name: ids for name, ids in settings["merged_classes"].items() if class_id in ids}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, settings["classes"], device)

    gts = {name: np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L")) for name in names}

    rows = []
    for value in values:
        thresholds = {class_id: value} if value > 0 else {}  # 0 = plain argmax, no threshold applied
        for name in names:
            pred = predict_mask(model, train_TIRF.resolve_image_path(settings["image_dir"], name), device, thresholds)
            gt = gts[name]
            g, p = gt == class_id, pred == class_id
            rows.append({"threshold": value, "class": args.class_name, **pixel_metrics_bool(g, p, cldice=False)})
            for merged_name, member_ids in merged_groups.items():
                g_u, p_u = np.isin(gt, member_ids), np.isin(pred, member_ids)
                rows.append({"threshold": value, "class": merged_name, **pixel_metrics_bool(g_u, p_u, cldice=False)})

    df = pd.DataFrame(rows)
    summary = df.groupby(["class", "threshold"])[["dice", "precision", "recall"]].mean().round(3)
    print(summary.to_string())

    out_path = args.out or os.path.join(run_dir, f"threshold_sweep_{args.class_name.replace(' ', '_')}.csv")
    summary.to_csv(out_path)
    print(f"\nSaved {out_path}")
    best = summary.xs(args.class_name, level="class")["dice"].idxmax()
    print(f"Best threshold for '{args.class_name}' by Dice: {best}")


if __name__ == "__main__":
    main()
