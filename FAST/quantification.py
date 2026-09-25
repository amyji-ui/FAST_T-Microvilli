#!/usr/bin/env python
# coding: utf-8
"""
Quantify the model's predicted masks and compare them with the manual masks.

For every image in the chosen subset (default: the held-out TEST images of a
training run) this script:
  1. runs the trained model to predict a class-id mask,
  2. compares it with the manual mask: per-class Dice, precision, recall, F1,
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
"""

import argparse
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

    class_names = {0: "Background"}
    for name, class_id in raw["class-name-to-index"].items():
        class_names[int(class_id)] = name
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
    settings["run_dir"] = run_dir or settings["out_dir"]
    return settings


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
    """Load run_dir/model/best_model.pth into a freshly built model."""
    weights = os.path.join(run_dir, "model", "best_model.pth")
    model = train_TIRF.build_model(classes, device)
    try:
        model.load_state_dict(torch.load(weights, map_location=device))
    except RuntimeError as error:
        raise SystemExit(
            f"{weights} does not fit a model with classes={classes}. It was probably trained with a "
            f"different class setup: point --run-dir at a run trained with the current config.\n"
            f"({str(error).splitlines()[0]})"
        )
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_path: str, device: torch.device) -> np.ndarray:
    """Predicted (H, W) uint8 class-id mask for one image, preprocessed
    exactly like validation/test images during training (grayscale ToTensor)."""
    image = Image.open(image_path).convert("L")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------

def pixel_metrics(gt: np.ndarray, pred: np.ndarray, class_id: int) -> dict:
    """Pixel-level agreement of one class between the manual (gt) and
    predicted mask. Undefined ratios (e.g. recall when the class is absent
    from the manual mask) are NaN and ignored when averaging over images."""
    g, p = gt == class_id, pred == class_id
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    nan = float("nan")
    return {
        "dice": 2 * tp / (g.sum() + p.sum()) if (g.sum() + p.sum()) else nan,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "gt_pixels": int(g.sum()),
        "pred_pixels": int(p.sum()),
    }


def count_objects(mask: np.ndarray, class_id: int, min_area: float) -> int:
    """Number of objects of a class: outer contours (cv2.findContours, as in
    the FAST paper) with area >= min_area. min_area=0 counts every object,
    including one-pixel-wide ones whose contour area is 0."""
    contours, _ = cv2.findContours((mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def class_fraction(mask: np.ndarray, class_id: int) -> float:
    """Class pixels / all foreground (non-background) pixels, the paper's
    definition of a class's fractional area."""
    foreground = int((mask != 0).sum())
    return float((mask == class_id).sum()) / foreground if foreground else float("nan")


def evaluate_images(model, names, split_of, settings, device, pred_dir):
    """Predict every image in `names` and compare with its manual mask.
    Returns (segmentation_df, quantification_df, {name: manual_mask},
    {name: predicted_mask}). Predicted masks are also saved to pred_dir."""
    class_names = settings["class_names"]
    foreground = [c for c in class_names if c != 0]
    seg_rows, quant_rows, gts, preds = [], [], {}, {}

    for name in names:
        gt = np.array(Image.open(os.path.join(settings["mask_dir"], name)).convert("L"))
        pred = predict_mask(model, os.path.join(settings["image_dir"], name), device)
        Image.fromarray(pred).save(os.path.join(pred_dir, name))
        gts[name], preds[name] = gt, pred
        common = {"image": name, "group": assign_group(name, settings["groups"]), "split": split_of.get(name, "not in split")}

        for class_id, class_name in class_names.items():
            seg_rows.append({**common, "class": class_name, **pixel_metrics(gt, pred, class_id)})

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
    """Dice, precision and recall per class (dots = images)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, metric in zip(axes, ["dice", "precision", "recall"]):
        values = {c: {g: seg_df[(seg_df["class"] == c) & (seg_df["group"] == g)][metric].to_numpy() for g in groups} for c in class_order}
        grouped_boxplot(ax, values, class_order, groups, [group_colors[g] for g in groups])
        ax.set_title(metric.capitalize())
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


def plot_overlays(names, image_dir, gts, preds, quant_df, colormap, class_names, counted_names, path) -> None:
    """Image | manual mask | predicted mask for each image, with object counts
    for the counted classes."""
    foreground = [c for c in class_names if c != 0 and class_names[c] in counted_names]
    fig, axes = plt.subplots(len(names), 3, figsize=(10.5, 3.5 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        image = np.array(Image.open(os.path.join(image_dir, name)).convert("L"))
        counts = quant_df[quant_df["image"] == name].set_index("class")

        def caption(kind):
            return " | ".join(f"{class_names[c]} {counts.loc[class_names[c], f'{kind}_count']}" for c in foreground)

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
    parser.add_argument("--max-overlays", type=int, default=6, help="Max images shown in the overlay figure.")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    run_dir = settings["run_dir"]
    out_dir = args.out_dir or os.path.join(run_dir, f"quantification_{args.subset}")
    pred_dir = os.path.join(out_dir, "predicted_masks")

    split_path = os.path.join(run_dir, "split.json")
    if not os.path.exists(split_path):
        raise SystemExit(f"{split_path} not found. train_TIRF.py writes it; retrain, or this run predates it.")
    with open(split_path) as f:
        split = json.load(f)
    split_of = {name: part for part, members in split.items() for name in members}
    names = sorted(f for f in os.listdir(settings["image_dir"]) if f.endswith(".png")) if args.subset == "all" else split[args.subset]
    missing = [n for n in names if not os.path.exists(os.path.join(settings["image_dir"], n)) or not os.path.exists(os.path.join(settings["mask_dir"], n))]
    if missing:
        raise SystemExit(f"{len(missing)} image/mask files listed for subset '{args.subset}' are missing, e.g. {missing[0]}")
    if not names:
        raise SystemExit(f"subset '{args.subset}' is empty")
    if args.subset != "test":
        print(f"NOTE: subset '{args.subset}' includes images used for training/validation; scores will be optimistic.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Quantifying {len(names)} '{args.subset}' images from {run_dir} on {device}")
    model = load_model(run_dir, settings["classes"], device)
    os.makedirs(pred_dir, exist_ok=True)  # only after everything above validated, so a bad --run-dir leaves nothing behind
    seg_df, quant_df, gts, preds = evaluate_images(model, names, split_of, settings, device, pred_dir)

    class_names = settings["class_names"]
    class_order = [class_names[c] for c in sorted(class_names)]
    foreground_names = [n for n in class_order if n != "Background"]
    group_order = list(settings["groups"]) + ["Unassigned"]
    seg_df = order_frame(seg_df, group_order, class_order)
    quant_df = order_frame(quant_df, group_order, class_order)
    groups = [g for g in group_order if (seg_df["group"] == g).any()]
    group_colors = {g: plt.cm.tab10(i) for i, g in enumerate(group_order)}  # stable colors when groups are added
    if "Unassigned" in groups:
        print(f"WARNING: {(seg_df['group'] == 'Unassigned').sum() // len(class_order)} image(s) match no group in config.yaml `groups`.")

    seg_summary = summarize(seg_df, ["dice", "precision", "recall", "f1"])
    quant_summary = summarize(quant_df, ["manual_count", "predicted_count", "count_ratio", "manual_fraction", "predicted_fraction"])
    comparison = compare_groups(quant_df)

    seg_df.to_csv(os.path.join(out_dir, "per_image_segmentation.csv"), index=False)
    quant_df.convert_dtypes().to_csv(os.path.join(out_dir, "per_image_quantification.csv"), index=False)  # whole-number counts, blank where not counted
    seg_summary.to_csv(os.path.join(out_dir, "segmentation_summary.csv"), index=False)
    quant_summary.to_csv(os.path.join(out_dir, "quantification_summary.csv"), index=False)
    if not comparison.empty:
        comparison.to_csv(os.path.join(out_dir, "group_comparison.csv"), index=False)

    plot_segmentation(seg_df, class_order, groups, group_colors, os.path.join(out_dir, "fig_segmentation.png"))
    counted_names = [n for n in foreground_names if n in settings["count_classes"]]
    if counted_names:
        plot_counts(quant_df, counted_names, groups, group_colors, os.path.join(out_dir, "fig_counts.png"))
    plot_group_comparison(quant_df, foreground_names, counted_names, groups, os.path.join(out_dir, "fig_group_comparison.png"))
    shown = names[: args.max_overlays]
    plot_overlays(shown, settings["image_dir"], gts, preds, quant_df, settings["colormap"], class_names, counted_names, os.path.join(out_dir, "fig_overlays.png"))

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
