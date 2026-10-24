#!/usr/bin/env python
"""Sweep per-class confidence thresholds and report test-set Dice for each value."""

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
    names = train_TIRF.test_split(current_filenames, settings["image_dir"])
    if not names:
        raise SystemExit("No test images found -- check that a dataset name ends in _Test and is converted.")
    print(f"Sweeping '{args.class_name}' threshold over {len(names)} test images from {run_dir}")

    merged_groups = {name: ids for name, ids in settings["merged_classes"].items() if class_id in ids}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, settings["classes"], device)

    gts = {name: np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L")) for name in names}

    rows = []
    for value in values:
        thresholds = {class_id: value} if value > 0 else {}
        for name in names:
            pred = predict_mask(model, train_TIRF.resolve_image_path(settings["image_dir"], name), device, thresholds)
            gt = gts[name]
            g, p = gt == class_id, pred == class_id
            group = train_TIRF.assign_group(name, settings["groups"])
            rows.append({"threshold": value, "group": group, "class": args.class_name, **pixel_metrics_bool(g, p, cldice=False)})
            for merged_name, member_ids in merged_groups.items():
                g_u, p_u = np.isin(gt, member_ids), np.isin(pred, member_ids)
                rows.append({"threshold": value, "group": group, "class": merged_name, **pixel_metrics_bool(g_u, p_u, cldice=False)})

    df = pd.DataFrame(rows)
    summary = df.groupby(["class", "threshold"])[["dice", "precision", "recall"]].mean().round(3)
    print(summary.to_string())

    by_group = df.groupby(["class", "group", "threshold"])[["dice", "precision", "recall"]].mean().round(3)
    for class_name in by_group.index.get_level_values("class").unique():
        print(f"\n{class_name} Dice per group (rows) by threshold (columns):")
        print(by_group.xs(class_name, level="class")["dice"].unstack("threshold").to_string())

    out_path = args.out or os.path.join(run_dir, f"threshold_sweep_{args.class_name.replace(' ', '_')}.csv")
    summary.to_csv(out_path)
    by_group.to_csv(out_path.replace(".csv", "_by_group.csv"))
    print(f"\nSaved {out_path} and {out_path.replace('.csv', '_by_group.csv')}")
    best = summary.xs(args.class_name, level="class")["dice"].idxmax()
    print(f"Best threshold for '{args.class_name}' by Dice: {best}")


if __name__ == "__main__":
    main()
