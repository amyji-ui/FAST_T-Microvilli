#!/usr/bin/env python
"""Per-cell morphological features from manual and predicted masks, plus a thesis summary table (run quantification.py first)."""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label as label_components, regionprops
from skimage.morphology import skeletonize

import train_TIRF
from quantification import count_objects_bool, load_settings
from thesis_figures import COMPARISON_ORDER, concordance, label

DENSITY_UNIT = 10000
FEATURES = [
    ("cell_area_px", "Cell footprint area (all labeled classes, holes filled)", "px", ""),
    ("cell_perimeter_px", "Perimeter of the largest connected footprint", "px", ""),
    ("cell_circularity", "4π·area / perimeter² of the largest footprint (1 = circle)", "–", "Sensitive to protrusions on the outline"),
    ("cell_solidity", "Footprint area / convex hull area of the largest footprint", "–", ""),
    ("cell_aspect_ratio", "Major / minor axis of the ellipse fitted to the largest footprint", "–", ""),
    ("body_area_px", "Cell body (lamellipodia) area", "px", ""),
    ("body_fraction", "Cell body area / cell area", "–", ""),
    ("factin_area_px", "F-actin area", "px", ""),
    ("factin_fraction", "F-actin area / cell area", "–", ""),
    ("factin_count", "Number of separate F-actin objects", "objects", "Unreliable: over-counts when a protrusion is predicted in pieces"),
    ("factin_length_px", "Total F-actin skeleton length", "px", ""),
    ("fa_area_px", "Focal adhesion area", "px", ""),
    ("fa_fraction", "Focal adhesion area / cell area", "–", ""),
    ("fa_count", "Number of separate focal adhesion objects", "objects", "Unreliable: aggregated microvilli are hard to separate"),
    ("microvilli_area_px", "Microvilli (F-actin + focal adhesion) area", "px", ""),
    ("microvilli_fraction", "Microvilli area / cell area (= F-actin fraction + focal adhesion fraction)", "–", "Primary microvilli measure"),
    ("microvilli_count", "Number of microvilli (connected objects of the union)", "objects", "Unreliable: touching microvilli merge, split predictions inflate it"),
    ("microvilli_density", f"Microvilli count per {DENSITY_UNIT:,} px of cell area", f"per {DENSITY_UNIT:,} px", "Unreliable: inherits the count errors; use microvilli fraction"),
    ("microvilli_length_px", "Total microvilli skeleton length", "px", ""),
    ("microvilli_mean_length_px", "Mean skeleton length per microvillus", "px", "Unreliable: depends on the count"),
    ("microvilli_mean_width_px", "Microvilli area / skeleton length", "px", ""),
    ("fa_share_of_microvilli", "Focal adhesion area / microvilli area (degree of aggregation)", "–", ""),
]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a boolean mask."""
    labeled = label_components(mask, connectivity=2)
    if labeled.max() == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def features(mask: np.ndarray, settings: dict) -> dict:
    """Every feature in FEATURES for one class-id mask."""
    name_to_id = {n: c for c, n in settings["class_names"].items()}
    factin, body, fa = mask == name_to_id["F-actin"], mask == name_to_id["Lamellipodia"], mask == name_to_id["Focal Adhesion"]
    microvilli = np.isin(mask, settings["merged_classes"]["microvilli"])
    min_area = settings["count_min_area"].get("microvilli", 0)

    footprint = ndi.binary_fill_holes(mask != 0)
    cell = int(footprint.sum())
    nan = float("nan")
    shape = {"cell_perimeter_px": nan, "cell_circularity": nan, "cell_solidity": nan, "cell_aspect_ratio": nan}
    if cell:
        props = regionprops(largest_component(footprint).astype(np.uint8))[0]
        shape = {"cell_perimeter_px": props.perimeter,
                 "cell_circularity": 4 * np.pi * props.area / props.perimeter ** 2 if props.perimeter else nan,
                 "cell_solidity": props.solidity,
                 "cell_aspect_ratio": props.major_axis_length / props.minor_axis_length if props.minor_axis_length else nan}

    def frac(x):
        return x.sum() / cell if cell else nan

    mv_count = count_objects_bool(microvilli, min_area)
    mv_length = int(skeletonize(microvilli).sum())
    return {
        "cell_area_px": cell, **shape,
        "body_area_px": int(body.sum()), "body_fraction": frac(body),
        "factin_area_px": int(factin.sum()), "factin_fraction": frac(factin),
        "factin_count": count_objects_bool(factin, 0), "factin_length_px": int(skeletonize(factin).sum()),
        "fa_area_px": int(fa.sum()), "fa_fraction": frac(fa), "fa_count": count_objects_bool(fa, 0),
        "microvilli_area_px": int(microvilli.sum()), "microvilli_fraction": frac(microvilli),
        "microvilli_count": mv_count,
        "microvilli_density": mv_count / cell * DENSITY_UNIT if cell else nan,
        "microvilli_length_px": mv_length,
        "microvilli_mean_length_px": mv_length / mv_count if mv_count else nan,
        "microvilli_mean_width_px": microvilli.sum() / mv_length if mv_length else nan,
        "fa_share_of_microvilli": fa.sum() / microvilli.sum() if microvilli.any() else nan,
    }


def summary_table(df: pd.DataFrame, groups: list) -> pd.DataFrame:
    """Per feature: definition, unit, note, model median [IQR] per group, and manual-vs-model CCC over all images."""
    rows = []
    for key, definition, unit, note in FEATURES:
        row = {"Feature": key, "Definition": definition, "Unit": unit}
        fmt = (lambda x: f"{x:,.0f}") if unit in ("px", "objects") and not key.endswith(("mean_length_px", "mean_width_px")) else \
              (lambda x: f"{x:.1f}") if unit != "–" else (lambda x: f"{x:.2f}")
        for g in groups:
            v = df.loc[df["group"] == g, f"predicted_{key}"].dropna()
            row[label(g, one_line=True)] = f"{fmt(v.median())} [{fmt(v.quantile(.25))}–{fmt(v.quantile(.75))}]" if len(v) else "–"
        pair = df[[f"manual_{key}", f"predicted_{key}"]].dropna()
        row["Manual vs model CCC"] = f"{concordance(pair.iloc[:, 0].to_numpy(float), pair.iloc[:, 1].to_numpy(float)):.2f}" if len(pair) > 2 else "–"
        row["Note"] = note
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder. Default: config's out-dir.")
    parser.add_argument("--quant-dir", default=None, help="quantification.py output. Default: <run-dir>/quantification")
    parser.add_argument("--out-dir", default="thesis_figures")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    quant_dir = args.quant_dir or os.path.join(settings["run_dir"], "quantification")
    pred_dir = os.path.join(quant_dir, "predicted_masks")
    images = pd.read_csv(os.path.join(quant_dir, "fraction_per_image.csv"))[["image", "group", "predicted_by"]]

    rows = []
    for _, r in images.iterrows():
        manual = np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], r["image"])).convert("L"))
        predicted = np.array(Image.open(os.path.join(pred_dir, r["image"])).convert("L"))
        rows.append({"image": r["image"], "group": r["group"], "predicted_by": r["predicted_by"],
                     **{f"manual_{k}": v for k, v in features(manual, settings).items()},
                     **{f"predicted_{k}": v for k, v in features(predicted, settings).items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(quant_dir, "features_per_image.csv"), index=False)

    groups = [g for g in COMPARISON_ORDER if g in set(df["group"])] + [g for g in settings["groups"] if g not in COMPARISON_ORDER and g in set(df["group"])]
    os.makedirs(os.path.join(args.out_dir, "comparison"), exist_ok=True)
    table = summary_table(df, groups)
    table.to_csv(os.path.join(args.out_dir, "comparison", "features.csv"), index=False, encoding="utf-8-sig")

    with pd.option_context("display.max_colwidth", 30, "display.width", 250):
        print(table[["Feature"] + [label(g, one_line=True) for g in groups] + ["Manual vs model CCC"]].to_string(index=False))
    print(f"\n{len(df)} images x {len(FEATURES)} features (manual and model) -> {os.path.join(quant_dir, 'features_per_image.csv')}")
    print(f"Table 4 -> {os.path.abspath(os.path.join(args.out_dir, 'comparison', 'features.csv'))}")


if __name__ == "__main__":
    main()
