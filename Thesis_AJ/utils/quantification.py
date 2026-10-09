#!/usr/bin/env python
# coding: utf-8
"""
Quantify the model's predicted masks and compare them with the manual masks.

For every image in the chosen subset (default: the held-out TEST images of a
training run) this script:
  1. runs the trained model to predict a class-id mask,
  2. compares it with the manual mask: per-class Dice, precision, recall, F1,
     and clDice,
  3. counts objects per class (OpenCV contours, as in the FAST paper) and
     computes each class's area fraction, for manual and predicted masks,
  4. summarises everything per group (CD8AA / CD8BB / Control ...) and tests
     for differences between groups,
  5. writes tables (CSV) and figures (PNG).

Groups come from filename patterns in config.yaml (`groups`), so adding the
CD8BB and Control images later needs no code change: put them in the labeled
project, reconvert, retrain, and rerun this script.

Usage (from the FAST folder):
    python quantification.py --config config.yaml --run-dir ../supervisely_run_3class

Note: at pixel level F1 equals Dice exactly (2TP / (2TP + FP + FN)). Both are
reported because both are commonly asked for; precision and recall are the
useful extra information (precision < recall = over-segmentation, and the
reverse = under-segmentation).

clDice (Shit et al., 2021) scores skeleton/connectivity agreement instead of
raw pixel overlap, so a boundary shift that doesn't break the filament's path
costs nothing -- it's the fairer metric for thin, filamentous classes like
F-actin, where plain Dice is dominated by sub-pixel boundary disagreement
rather than real segmentation errors. It's NaN for Background (meaningless
for a non-filamentous fill class).
"""

import argparse
import glob
import json
import os
from itertools import combinations
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
from PIL import Image
from scipy import stats
from skimage.morphology import skeletonize
from torchvision import transforms

import train_TIRF  # reuse build_model / load_config so the setup stays defined in one place


# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------

def load_settings(config_path: str, run_dir: Optional[str]) -> dict:
    """Resolve settings from config.yaml: the training settings (image/mask
    dirs, classes, colormap) via train_TIRF.load_config, plus this script's
    own keys (class-name-to-index, groups, count-min-area).

    `run_dir` (a training output folder holding model/best_model.pth and
    split.json) defaults to the config's out-dir.
    """
    train_args = train_TIRF.build_arg_parser().parse_args(["--config", config_path])
    settings = train_TIRF.load_config(train_args)
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    settings["project_dir"] = raw.get("project-dir")
    class_names = {0: "Background"}
    for name, class_id in raw["class-name-to-index"].items():
        # First name listed per id is the canonical display name; later ones
        # mapping to the same id (e.g. "F-actin_1") are aliases only.
        class_names.setdefault(int(class_id), name)
    if sorted(class_names) != list(range(settings["classes"])):
        raise ValueError("class-name-to-index must define class ids 1..classes-1 (background is 0)")

    settings["class_names"] = class_names
    settings["groups"] = raw.get("groups") or {}
    settings["count_min_area"] = raw.get("count-min-area") or {}

    foreground_names = [n for c, n in class_names.items() if c != 0]
    count_classes = raw.get("count-classes")
    count_classes = foreground_names if count_classes is None else list(count_classes)
    unknown = [n for n in count_classes if n not in foreground_names]
    if unknown:
        raise ValueError(f"count-classes lists unknown class(es) {unknown}; foreground classes are {foreground_names}")
    settings["count_classes"] = count_classes

    # Derived "merged" pseudo-classes for quantification only (see
    # merged-classes in config.yaml): {name: [class_id, ...]}, resolved from
    # name to id here once, same as class_name_to_index above.
    name_to_id = {name: cid for cid, name in class_names.items()}
    raw_merged = raw.get("merged-classes") or {}
    merged_classes = {}
    for merged_name, member_names in raw_merged.items():
        unknown = [n for n in member_names if n not in name_to_id]
        if unknown:
            raise ValueError(f"merged-classes '{merged_name}' lists unknown class(es) {unknown}; known classes are {list(name_to_id)}")
        merged_classes[merged_name] = [name_to_id[n] for n in member_names]
    settings["merged_classes"] = merged_classes

    # Default per-class confidence thresholds (config.yaml's class-thresholds
    # and threshold-fallback) -- see predict_mask and --class-threshold,
    # which override these per-class on the command line for one run.
    raw_thresholds = raw.get("class-thresholds") or {}
    unknown = [n for n in raw_thresholds if n not in name_to_id]
    if unknown:
        raise ValueError(f"class-thresholds lists unknown class(es) {unknown}; known classes are {list(name_to_id)}")
    settings["class_thresholds"] = {name_to_id[n]: float(v) for n, v in raw_thresholds.items()}
    settings["threshold_fallback"] = raw.get("threshold-fallback", "next_best")

    settings["run_dir"] = run_dir or settings["out_dir"]
    return settings


def parse_class_thresholds(raw_args: list, class_names: dict) -> dict:
    """Parses --class-threshold's repeated "CLASS=VALUE" strings into
    {class_id: float}, using class_names (settings["class_names"]) to
    resolve a class name to its id. See predict_mask."""
    name_to_id = {name: cid for cid, name in class_names.items()}
    thresholds = {}
    for raw in raw_args:
        if "=" not in raw:
            raise SystemExit(f"--class-threshold expects CLASS=VALUE, got '{raw}'")
        name, value = raw.split("=", 1)
        if name not in name_to_id:
            raise SystemExit(f"--class-threshold: unknown class '{name}'; known classes are {list(name_to_id)}")
        try:
            thresholds[name_to_id[name]] = float(value)
        except ValueError:
            raise SystemExit(f"--class-threshold: '{value}' is not a number (from '{raw}')")
    return thresholds


def assign_group(filename: str, groups: dict) -> str:
    """First group in `groups` with a pattern contained in the filename
    (case-insensitive) wins; otherwise "Unassigned"."""
    lowered = filename.lower()
    for group, patterns in groups.items():
        if any(str(p).lower() in lowered for p in patterns):
            return group
    return "Unassigned"


# --------------------------------------------------------------------------
# 2. Prediction
# --------------------------------------------------------------------------

def load_model(run_dir: str, classes: int, device: torch.device) -> torch.nn.Module:
    """Load run_dir/model/ into an EnsembleModel averaging every k-fold CV
    fold's checkpoint (model/fold*_model.pth), instead of trusting a single
    best-val-loss pick -- see train_TIRF.EnsembleModel's docstring for why.
    Falls back to a single model/best_model.pth for an older run that
    predates per-fold saving."""
    model_dir = os.path.join(run_dir, "model")
    fold_paths = sorted(glob.glob(os.path.join(model_dir, "fold*_model.pth")))
    weights_paths = fold_paths or [os.path.join(model_dir, "best_model.pth")]

    members = []
    for weights in weights_paths:
        member = train_TIRF.build_model(classes, device)
        try:
            member.load_state_dict(torch.load(weights, map_location=device))
        except RuntimeError as error:
            raise SystemExit(
                f"{weights} does not fit a model with classes={classes}. It was probably trained with a "
                f"different class setup: point --run-dir at a run trained with the current config.\n"
                f"({str(error).splitlines()[0]})"
            )
        members.append(member)

    ensemble = train_TIRF.EnsembleModel(members).to(device)
    ensemble.eval()
    return ensemble


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_path: str, device: torch.device,
                  class_thresholds: Optional[dict] = None, threshold_fallback: str = "next_best") -> np.ndarray:
    """Predicted (H, W) uint8 class-id mask for one image, preprocessed
    exactly like validation/test images during training (grayscale ToTensor).

    `model` is expected to return per-class PROBABILITIES, not logits --
    true for every model load_model returns, since it always wraps in
    train_TIRF.EnsembleModel (softmaxes internally), even for a single
    checkpoint.

    class_thresholds: optional {class_id: min_confidence}. Without it,
    plain argmax (whichever class scores highest for a pixel wins, as
    always). With it, a class in the dict only wins a pixel if its
    probability ALSO clears its threshold -- raising its threshold shrinks
    its predicted region without retraining, the fix for a class whose
    predictions are systematically too thick (see sweep_threshold.py).

    threshold_fallback controls what happens to a pixel that WOULD have won
    for a thresholded class (it was the plain-argmax winner) but didn't
    clear that class's bar:
      "next_best" (default): falls through to the best REMAINING class
          (never forced to Background). Can bleed into a neighboring class
          that's spatially common nearby (e.g. demoted F-actin pixels
          landing on Focal Adhesion), inflating ITS count/area as a
          side effect -- fine if that class's count isn't what you're
          measuring (e.g. a merged-classes group containing both absorbs
          the bleed with no net effect), not fine if you need that
          neighboring class's own numbers to stay clean.
      "background": forced straight to Background instead, with no
          fallback competition -- cleaner isolation from neighboring
          classes, at the cost of not "rescuing" a pixel that was probably
          still something, just not confidently enough that class.
    """
    image = Image.open(image_path).convert("L")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    probs = model(tensor)[0]  # (C, H, W) probabilities

    if not class_thresholds:
        return probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

    original_argmax = probs.argmax(dim=0)
    probs = probs.clone()
    fail_masks = {}
    for class_id, threshold in class_thresholds.items():
        fail_masks[class_id] = probs[class_id] < threshold
        probs[class_id] = torch.where(fail_masks[class_id], torch.full_like(probs[class_id], -1.0), probs[class_id])
    result = probs.argmax(dim=0)

    if threshold_fallback == "background":
        demoted = torch.zeros_like(original_argmax, dtype=torch.bool)
        for class_id, fails in fail_masks.items():
            demoted |= (original_argmax == class_id) & fails
        result = torch.where(demoted, torch.zeros_like(result), result)
    elif threshold_fallback != "next_best":
        raise ValueError(f"threshold_fallback must be 'next_best' or 'background', got '{threshold_fallback}'")

    return result.cpu().numpy().astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------

def cl_dice(g: np.ndarray, p: np.ndarray) -> float:
    """clDice (Shit et al., 2021): Dice between each mask's 1px-wide skeleton
    and the OTHER mask's full region, instead of between the raw regions. A
    boundary/width disagreement that still contains the correct skeleton path
    costs nothing, so this scores whether the filament's path and
    connectivity were found, independent of exact edge placement -- unlike
    plain pixel Dice, which is dominated by boundary noise for thin classes.
    NaN if either mask has no pixels of this class (undefined, as for the
    other ratios in pixel_metrics)."""
    skel_g, skel_p = skeletonize(g), skeletonize(p)
    if not skel_g.any() or not skel_p.any():
        return float("nan")
    t_prec = (skel_p & g).sum() / skel_p.sum()  # predicted skeleton inside manual region
    t_sens = (skel_g & p).sum() / skel_g.sum()  # manual skeleton inside predicted region
    return 2 * t_prec * t_sens / (t_prec + t_sens) if (t_prec + t_sens) else float("nan")


def pixel_metrics_bool(g: np.ndarray, p: np.ndarray, cldice: bool = True) -> dict:
    """Pixel-level agreement between two boolean masks (manual g, predicted
    p) -- a single real class's pixels, or (see merged-classes in
    config.yaml) the union of several classes' pixels treated as one group.
    Undefined ratios (e.g. recall when the class/group is absent from the
    manual mask) are NaN and ignored when averaging over images. Pass
    cldice=False to skip it (e.g. Background: not a filamentous class)."""
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    nan = float("nan")
    return {
        "dice": 2 * tp / (g.sum() + p.sum()) if (g.sum() + p.sum()) else nan,
        "cldice": cl_dice(g, p) if cldice else nan,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "gt_pixels": int(g.sum()),
        "pred_pixels": int(p.sum()),
    }


def pixel_metrics(gt: np.ndarray, pred: np.ndarray, class_id: int) -> dict:
    """Pixel-level agreement of one class (see pixel_metrics_bool). clDice
    skipped for class_id 0 (Background): not a filamentous class."""
    return pixel_metrics_bool(gt == class_id, pred == class_id, cldice=class_id != 0)


def count_objects_bool(bool_mask: np.ndarray, min_area: float) -> int:
    """Number of objects in a boolean mask: outer contours (cv2.findContours,
    as in the FAST paper) with area >= min_area. min_area=0 counts every
    object, including one-pixel-wide ones whose contour area is 0."""
    contours, _ = cv2.findContours(bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def count_objects(mask: np.ndarray, class_id: int, min_area: float) -> int:
    """Number of objects of one class (see count_objects_bool)."""
    return count_objects_bool(mask == class_id, min_area)


def fraction_of_foreground(mask: np.ndarray, member_mask: np.ndarray) -> float:
    """member_mask's pixel count / all foreground (non-background) pixels in
    the full multi-class `mask`, the paper's definition of a class's
    fractional area -- generalized to any boolean subset of `mask`, not just
    a single class_id (see class_fraction, merged-classes in config.yaml)."""
    foreground = int((mask != 0).sum())
    return float(member_mask.sum()) / foreground if foreground else float("nan")


def class_fraction(mask: np.ndarray, class_id: int) -> float:
    """Class pixels / all foreground (non-background) pixels (see
    fraction_of_foreground)."""
    return fraction_of_foreground(mask, mask == class_id)


def evaluate_images(model, names, split_of, settings, device, pred_dir, class_thresholds=None, threshold_fallback="next_best"):
    """Predict every image in `names` and compare with its manual mask.
    Returns (segmentation_df, quantification_df, {name: manual_mask},
    {name: predicted_mask}). Predicted masks are also saved to pred_dir.
    class_thresholds/threshold_fallback: see predict_mask."""
    class_names = settings["class_names"]
    foreground = [c for c in class_names if c != 0]
    seg_rows, quant_rows, gts, preds = [], [], {}, {}

    for name in names:
        gt = np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L"))
        pred = predict_mask(model, train_TIRF.resolve_image_path(settings["image_dir"], name), device, class_thresholds, threshold_fallback)
        pred_path = os.path.join(pred_dir, name)  # name is "<dataset>/<filename>.png" -- mirrors the dataset structure here too
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        Image.fromarray(pred).save(pred_path)
        gts[name], preds[name] = gt, pred
        common = {"image": name, "group": assign_group(name, settings["groups"]), "split": split_of.get(name, "not in split")}

        # Background excluded entirely: it's not a class anyone cares about
        # measuring (its Dice is trivially ~1 given how much of the image is
        # background, and it was never counted or clDice'd anyway).
        for class_id in foreground:
            seg_rows.append({**common, "class": class_names[class_id], **pixel_metrics(gt, pred, class_id)})

        for class_id in foreground:
            class_name = class_names[class_id]
            counted = class_name in settings["count_classes"]  # uncounted classes get NaN counts, but still get fractions
            min_area = settings["count_min_area"].get(class_name, 0)
            manual_count = count_objects(gt, class_id, min_area) if counted else float("nan")
            predicted_count = count_objects(pred, class_id, min_area) if counted else float("nan")
            quant_rows.append({
                **common,
                "class": class_name,
                "manual_count": manual_count,
                "predicted_count": predicted_count,
                "count_ratio": predicted_count / manual_count if counted and manual_count else float("nan"),
                "manual_fraction": class_fraction(gt, class_id),
                "predicted_fraction": class_fraction(pred, class_id),
            })

        # Merged pseudo-classes (config.yaml's merged-classes, e.g. microvilli
        # = F-actin + Focal Adhesion, since both are considered part of the
        # same underlying structure): segmentation quality (Dice/clDice/
        # precision/recall) AND count+area, all computed on the UNION mask
        # -- so a Focal Adhesion blob touching an F-actin region is one
        # microvillus for counting, and pixels of either class count as a
        # "hit" for Dice/clDice, not just an exact class match. The
        # per-real-class rows above are unaffected, so it's still possible
        # to tell which class drives an error when something looks off.
        for merged_name, member_ids in settings["merged_classes"].items():
            gt_union = np.isin(gt, member_ids)
            pred_union = np.isin(pred, member_ids)

            seg_rows.append({**common, "class": merged_name, **pixel_metrics_bool(gt_union, pred_union)})

            min_area = settings["count_min_area"].get(merged_name, 0)
            manual_count = count_objects_bool(gt_union, min_area)
            predicted_count = count_objects_bool(pred_union, min_area)
            quant_rows.append({
                **common,
                "class": merged_name,
                "manual_count": manual_count,
                "predicted_count": predicted_count,
                "count_ratio": predicted_count / manual_count if manual_count else float("nan"),
                "manual_fraction": fraction_of_foreground(gt, gt_union),
                "predicted_fraction": fraction_of_foreground(pred, pred_union),
            })

    return pd.DataFrame(seg_rows), pd.DataFrame(quant_rows), gts, preds


# --------------------------------------------------------------------------
# 4. Summaries and group comparison
# --------------------------------------------------------------------------

def order_frame(df: pd.DataFrame, group_order: list, class_order: list) -> pd.DataFrame:
    """Sort rows by group (config order) then class, and make both columns
    ordered categoricals so tables and plots keep that order."""
    df = df.copy()
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df["class"] = pd.Categorical(df["class"], categories=class_order, ordered=True)
    return df.sort_values(["group", "class", "image"]).reset_index(drop=True)


def summarize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Mean and standard deviation over images of each column, per group and class."""
    grouped = df.groupby(["group", "class"], observed=True)
    out = grouped[columns].agg(["mean", "std"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out.insert(0, "n_images", grouped["image"].nunique())
    return out.reset_index()


def compare_groups(quant_df: pd.DataFrame) -> pd.DataFrame:
    """Test whether groups differ in each class's counts and fractions
    (manual and predicted): Kruskal-Wallis across all groups when there are
    three or more, and Mann-Whitney U for every pair of groups. Groups need at
    least 2 images. No multiple-testing correction, and with few images per
    group the p-values are only indicative."""
    metrics = ["manual_count", "predicted_count", "manual_fraction", "predicted_fraction"]
    rows = []
    for class_name, sub in quant_df.groupby("class", observed=True):
        for metric in metrics:
            samples = {g: s[metric].dropna().to_numpy() for g, s in sub.groupby("group", observed=True)}
            samples = {g: v for g, v in samples.items() if len(v) >= 2}
            if len(samples) >= 3:
                try:
                    statistic, p_value = stats.kruskal(*samples.values())
                except ValueError:  # every value identical
                    statistic, p_value = float("nan"), float("nan")
                rows.append({"class": class_name, "metric": metric, "comparison": "all groups", "test": "Kruskal-Wallis",
                             "n_a": sum(len(v) for v in samples.values()), "n_b": "", "statistic": statistic, "p_value": p_value})
            for a, b in combinations(samples, 2):
                statistic, p_value = stats.mannwhitneyu(samples[a], samples[b], alternative="two-sided")
                rows.append({"class": class_name, "metric": metric, "comparison": f"{a} vs {b}", "test": "Mann-Whitney U",
                             "n_a": len(samples[a]), "n_b": len(samples[b]), "statistic": statistic, "p_value": p_value})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------

def grouped_boxplot(ax, values: dict, x_labels: list, hues: list, colors: list) -> None:
    """Boxplots at each x label, one box per hue side by side, with the
    individual images drawn as dots. values[x][hue] is an array."""
    width = 0.8 / len(hues)
    rng = np.random.default_rng(0)
    for i, x in enumerate(x_labels):
        for j, hue in enumerate(hues):
            data = np.asarray(values[x][hue], dtype=float)
            data = data[~np.isnan(data)]
            if data.size == 0:
                continue
            position = i + (j - (len(hues) - 1) / 2) * width
            box = ax.boxplot([data], positions=[position], widths=width * 0.85, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.55)
            for median in box["medians"]:
                median.set_color("black")
            ax.scatter(position + rng.uniform(-width * 0.25, width * 0.25, data.size), data, s=14, color="black", zorder=3)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)


def legend_handles(labels: list, colors: list) -> list:
    return [Patch(facecolor=c, alpha=0.55, label=l) for l, c in zip(labels, colors)]


def plot_segmentation(seg_df, class_order, groups, group_colors, path) -> None:
    """Dice, clDice, precision and recall per class (dots = images)."""
    metrics = ["dice", "cldice", "precision", "recall"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.6), sharey=True)
    for ax, metric in zip(axes, metrics):
        values = {c: {g: seg_df[(seg_df["class"] == c) & (seg_df["group"] == g)][metric].to_numpy() for g in groups} for c in class_order}
        grouped_boxplot(ax, values, class_order, groups, [group_colors[g] for g in groups])
        ax.set_title("clDice" if metric == "cldice" else metric.capitalize())
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score (1 = perfect)")
    axes[0].legend(handles=legend_handles(groups, [group_colors[g] for g in groups]), title="Group", loc="lower left")
    fig.suptitle("Segmentation quality: predicted vs manual mask (each dot = one image)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_counts(quant_df, foreground_names, groups, group_colors, path) -> None:
    """Manual vs predicted object counts per foreground class."""
    fig, axes = plt.subplots(1, len(foreground_names), figsize=(5.4 * len(foreground_names), 4.8), squeeze=False)
    for ax, class_name in zip(axes[0], foreground_names):
        sub = quant_df[quant_df["class"] == class_name]
        for g in groups:
            s = sub[sub["group"] == g]
            ax.scatter(s["manual_count"], s["predicted_count"], label=g, color=group_colors[g], s=42, edgecolor="black", linewidth=0.5, zorder=3)
        all_counts = sub[["manual_count", "predicted_count"]].to_numpy()
        top = max(float(all_counts.max()), 1) * 1.4
        if (all_counts > 0).all():  # log-log like the paper; same limits on both axes so y = x is the diagonal
            bottom = float(all_counts.min()) / 1.4
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:  # zero counts can't be shown on a log axis
            bottom = 0
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
        ax.plot([bottom, top], [bottom, top], "k--", lw=1, label="predicted = manual")
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        if bottom > 0:  # readable tick labels on the log axes
            ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000) if bottom <= t <= top]
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator(ticks))
                axis.set_major_formatter(ScalarFormatter())
                axis.set_minor_locator(NullLocator())
        note = [f"n = {len(sub)}"]
        if len(sub) >= 3 and sub["manual_count"].nunique() > 1 and sub["predicted_count"].nunique() > 1:
            note.append(f"Pearson r = {np.corrcoef(sub['manual_count'], sub['predicted_count'])[0, 1]:.2f}")
        if sub["count_ratio"].notna().any():
            note.append(f"median predicted / manual = {sub['count_ratio'].median():.1f}x")
        ax.text(0.04, 0.96, "\n".join(note), transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(class_name)
        ax.set_xlabel("manual count")
        ax.set_ylabel("predicted count")
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Object counts per image (contours): manual vs predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_comparison(quant_df, foreground_names, counted_names, groups, path) -> None:
    """Fraction and count per group, manual next to predicted, one panel per
    class. Count panels are left empty for classes that are not counted."""
    rows = [("fraction", "fraction of foreground area"), ("count", "objects per image")]
    fig, axes = plt.subplots(len(rows), len(foreground_names), figsize=(5.4 * len(foreground_names), 8.4), squeeze=False)
    sources = ["Manual", "Predicted"]
    for r, (metric, ylabel) in enumerate(rows):
        for c, class_name in enumerate(foreground_names):
            sub = quant_df[quant_df["class"] == class_name]
            values = {g: {"Manual": sub[sub["group"] == g][f"manual_{metric}"].to_numpy(),
                          "Predicted": sub[sub["group"] == g][f"predicted_{metric}"].to_numpy()} for g in groups}
            ax = axes[r][c]
            if metric == "count" and class_name not in counted_names:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{class_name}: not counted\n(see count-classes in config.yaml)", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            grouped_boxplot(ax, values, groups, sources, ["lightgray", "tab:blue"])
            ax.set_title(f"{class_name}: {metric}")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(handles=legend_handles(sources, ["lightgray", "tab:blue"]), loc="best")
    fig.suptitle("Comparison between groups (each dot = one image)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlays(names, image_dir, gts, preds, quant_df, colormap, counted_names, path) -> None:
    """Image | manual mask | predicted mask for each image, with object counts
    for the counted classes (real or merged -- counted_names is name
    strings, not class ids, since a merged-classes group like "microvilli"
    has no single id of its own)."""
    fig, axes = plt.subplots(len(names), 3, figsize=(10.5, 3.5 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        image = np.array(Image.open(train_TIRF.resolve_image_path(image_dir, name)).convert("L"))
        counts = quant_df[quant_df["image"] == name].set_index("class")

        def caption(kind):
            return " | ".join(f"{cname} {counts.loc[cname, f'{kind}_count']}" for cname in counted_names)

        row[0].imshow(image, cmap="gray", vmax=np.percentile(image, 99.5))
        row[0].set_title(name[-30:], fontsize=8)
        row[1].imshow(colormap[gts[name]])
        row[1].set_title("Manual\n" + caption("manual"), fontsize=8)
        row[2].imshow(colormap[preds[name]])
        row[2].set_title("Predicted\n" + caption("predicted"), fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. Entrypoint
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder (model/best_model.pth + split.json). Default: config's out-dir.")
    parser.add_argument("--subset", choices=["test", "val", "train", "all"], default="test",
                        help="Which images to quantify. Use test for honest numbers: train/all include images the model was trained on.")
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/quantification_<subset>")
    parser.add_argument("--overlays-per-fig", type=int, default=4, help="Images per overlay figure. Every image in the subset is shown, split across fig_overlays.png, fig_overlays_2.png, ...")
    parser.add_argument("--class-threshold", action="append", default=[], metavar="CLASS=VALUE",
                         help="Override config.yaml's class-thresholds for one class, for one run. Require this class's "
                              "softmax probability to clear VALUE before it's allowed to win a pixel -- raises the bar "
                              "for a class whose predicted region is systematically too thick, instead of plain argmax. "
                              "Repeatable, e.g. --class-threshold F-actin=0.65. See sweep_threshold.py to find a good "
                              "value first.")
    parser.add_argument("--threshold-fallback", choices=["next_best", "background"], default=None,
                         help="Overrides config.yaml's threshold-fallback for one run: what a pixel demoted by a class "
                              "threshold becomes, the best remaining class (can bleed into a spatially-common neighbor, "
                              "inflating ITS count as a side effect) or Background (cleaner isolation, no rescue). See "
                              "predict_mask's docstring.")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    run_dir = settings["run_dir"]
    out_dir = args.out_dir or os.path.join(run_dir, f"quantification_{args.subset}")
    pred_dir = os.path.join(out_dir, "predicted_masks")

    current_filenames = train_TIRF.list_dataset_images(settings["image_dir"])

    split_path = os.path.join(run_dir, "split.json")
    split = {}
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)

    if "folds" in split:
        # Ensembled run (see train_TIRF.EnsembleModel): every pool image is
        # val in exactly one fold and train in the rest, so there's no
        # single held-out val set anymore -- "train" and "val" both resolve
        # to the full non-test pool the ensemble has substantially trained
        # on (union across folds).
        pool_names = sorted({n for fold in split["folds"] for n in fold["train"] + fold["val"]})
        split_of = {n: "train+val (ensemble pool)" for n in pool_names}
        split_of.update({n: "test" for n in split.get("test", [])})
    else:
        split_of = {name: part for part, members in split.items() if part in ("train", "val", "test") for name in members}

    if args.subset == "all":
        names = current_filenames
    elif args.subset == "test":
        # Always freshly determined from test-source-dir (see train_TIRF.py's
        # resolve_pinned_test), never from this run's split.json -- so this
        # works correctly even for a run trained before Labeled_converted was
        # last regenerated, or before a later test image was added. Test
        # membership is a config-level fact, not something tied to one run.
        names = train_TIRF.resolve_pinned_test(current_filenames, settings["image_dir"], settings["test_source_dir"])
    elif "folds" in split:
        if args.subset not in ("train", "val"):
            raise SystemExit(f"subset '{args.subset}' not found in {split_path}")
        names = sorted({n for fold in split["folds"] for n in fold[args.subset]})
    else:
        if not split:
            raise SystemExit(f"{split_path} not found or empty. train_TIRF.py writes it; retrain, or this run predates it.")
        if args.subset not in split:
            raise SystemExit(f"subset '{args.subset}' not found in {split_path}")
        names = split[args.subset]

    missing = [n for n in names if not os.path.exists(train_TIRF.resolve_image_path(settings["image_dir"], n)) or not os.path.exists(train_TIRF.resolve_mask_path(settings["mask_dir"], n))]
    if missing:
        raise SystemExit(f"{len(missing)} image/mask files listed for subset '{args.subset}' are missing, e.g. {missing[0]}")
    if not names:
        raise SystemExit(f"subset '{args.subset}' is empty")
    if args.subset != "test":
        print(f"NOTE: subset '{args.subset}' includes images used for training/validation; scores will be optimistic.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Quantifying {len(names)} '{args.subset}' images from {run_dir} on {device}")
    # config.yaml's class-thresholds, with --class-threshold overriding a
    # single class's value for this run only.
    class_thresholds = {**settings["class_thresholds"], **parse_class_thresholds(args.class_threshold, settings["class_names"])}
    threshold_fallback = args.threshold_fallback if args.threshold_fallback is not None else settings["threshold_fallback"]
    if class_thresholds:
        print(f"Class thresholds: { {settings['class_names'][c]: v for c, v in class_thresholds.items()} } (fallback: {threshold_fallback})")

    model = load_model(run_dir, settings["classes"], device)
    os.makedirs(pred_dir, exist_ok=True)  # only after everything above validated, so a bad --run-dir leaves nothing behind
    seg_df, quant_df, gts, preds = evaluate_images(model, names, split_of, settings, device, pred_dir, class_thresholds, threshold_fallback)

    class_names = settings["class_names"]
    # Background is excluded entirely (see evaluate_images) -- not a class
    # anyone measures. Both seg_df and quant_df have one row per
    # merged-classes group (e.g. "microvilli") per image too, alongside the
    # real per-class rows, so class_order needs to include merged group
    # names as well.
    merged_names = list(settings["merged_classes"])
    class_order = [class_names[c] for c in sorted(class_names) if c != 0] + merged_names
    foreground_names = class_order
    group_order = list(settings["groups"]) + ["Unassigned"]
    seg_df = order_frame(seg_df, group_order, class_order)
    quant_df = order_frame(quant_df, group_order, class_order)
    groups = [g for g in group_order if (seg_df["group"] == g).any()]
    group_colors = {g: plt.cm.tab10(i) for i, g in enumerate(group_order)}  # stable colors when groups are added
    if "Unassigned" in groups:
        print(f"WARNING: {(seg_df['group'] == 'Unassigned').sum() // len(class_order)} image(s) match no group in config.yaml `groups`.")

    seg_summary = summarize(seg_df, ["dice", "cldice", "precision", "recall", "f1"])
    quant_summary = summarize(quant_df, ["manual_count", "predicted_count", "count_ratio", "manual_fraction", "predicted_fraction"])
    comparison = compare_groups(quant_df)

    seg_df.to_csv(os.path.join(out_dir, "per_image_segmentation.csv"), index=False)
    quant_df.convert_dtypes().to_csv(os.path.join(out_dir, "per_image_quantification.csv"), index=False)  # whole-number counts, blank where not counted
    seg_summary.to_csv(os.path.join(out_dir, "segmentation_summary.csv"), index=False)
    quant_summary.to_csv(os.path.join(out_dir, "quantification_summary.csv"), index=False)
    if not comparison.empty:
        comparison.to_csv(os.path.join(out_dir, "group_comparison.csv"), index=False)

    plot_segmentation(seg_df, class_order, groups, group_colors, os.path.join(out_dir, "fig_segmentation.png"))
    # Merged-classes groups (e.g. microvilli) are always counted -- that's
    # their whole purpose -- independent of config.yaml's count-classes,
    # which only governs the real per-class rows.
    counted_names = [n for n in foreground_names if n in settings["count_classes"]] + merged_names
    if counted_names:
        plot_counts(quant_df, counted_names, groups, group_colors, os.path.join(out_dir, "fig_counts.png"))
    plot_group_comparison(quant_df, foreground_names, counted_names, groups, os.path.join(out_dir, "fig_group_comparison.png"))
    for old in glob.glob(os.path.join(out_dir, "fig_overlays*.png")):
        os.remove(old)
    chunk = args.overlays_per_fig
    for i in range(0, len(names), chunk):
        group = names[i : i + chunk]
        suffix = "" if i == 0 else f"_{i // chunk + 1}"
        plot_overlays(group, settings["image_dir"], gts, preds, quant_df, settings["colormap"], counted_names, os.path.join(out_dir, f"fig_overlays{suffix}.png"))

    fmt = lambda v: f"{v:.3f}"
    print("\n=== Segmentation quality (mean over images) ===")
    print(seg_summary[["group", "class", "n_images"] + [c for c in seg_summary if c.endswith("_mean")]]
          .rename(columns=lambda c: c.replace("_mean", "")).to_string(index=False, float_format=fmt))
    print("\n=== Manual vs predicted (mean over images) ===")
    print(quant_summary[["group", "class", "n_images", "manual_count_mean", "predicted_count_mean", "count_ratio_mean",
                         "manual_fraction_mean", "predicted_fraction_mean"]].to_string(index=False, float_format=fmt, na_rep="-"))
    if comparison.empty:
        print("\nGroup comparison skipped: it needs at least two groups with two or more images each.")
    else:
        print(f"\nGroup comparison tests saved to group_comparison.csv ({len(comparison)} rows).")
    print(f"\nAll tables and figures saved to {out_dir}")


if __name__ == "__main__":
    main()
