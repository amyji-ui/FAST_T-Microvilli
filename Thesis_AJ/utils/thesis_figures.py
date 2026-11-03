#!/usr/bin/env python
"""Thesis figures and tables from a quantification run (run quantification.py first)."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from PIL import Image
from scipy import stats

import train_TIRF
from quantification import (INK, INK_SECONDARY, MANUAL_COLOR, MODEL_COLOR, box_with_dots, holm,
                            load_model, load_settings, predict_mask, style_axis)

LABELS = {
    "Control": "Control",
    "CD8A_GFP": "CD8A\nGFP",
    "CD8A_ABS": "CD8A\nAb",
    "CD8A_Trunc_GFP": "Trunc CD8A\nGFP",
    "CD8A_Trunc_ABS": "Trunc CD8A\nAb",
    "CD8AA_old": "CD8AA\n(earlier batch)",
}
MASK_COLORS = {0: "#ffffff", 1: "#2a78d6", 2: "#d9d8d4", 3: "#eb6834"}
CLASS_LABELS = {"Lamellipodia": "Cell body (lamellipodia)", "Focal Adhesion": "Focal adhesion",
                "microvilli": "Microvilli (F-actin + focal adhesion)"}
MASK_LEGEND = [(2, "Cell body (lamellipodia)"), (1, "F-actin"), (3, "Focal adhesion")]
COMPARISON_ORDER = ["Control", "CD8A_GFP", "CD8A_Trunc_GFP", "CD8A_ABS", "CD8A_Trunc_ABS"]
CONTRASTS = [
    ("CD8A vs Control", "Control", "CD8A_GFP"),
    ("CD8A vs Control", "Control", "CD8A_Trunc_GFP"),
    ("CD8A vs Control", "Control", "CD8A_ABS"),
    ("CD8A vs Control", "Control", "CD8A_Trunc_ABS"),
    ("Truncation (Trunc - Full)", "CD8A_GFP", "CD8A_Trunc_GFP"),
    ("Truncation (Trunc - Full)", "CD8A_ABS", "CD8A_Trunc_ABS"),
    ("Detection (Ab - GFP)", "CD8A_GFP", "CD8A_ABS"),
    ("Detection (Ab - GFP)", "CD8A_Trunc_GFP", "CD8A_Trunc_ABS"),
]
SOURCES = [("manual", "Manual", MANUAL_COLOR), ("predicted", "Model", MODEL_COLOR)]
COMPARE_METRIC = "factin_fraction"
COMPARE_LABEL = "F-actin fraction of cell area"
COMPARE_SHORT = "F-actin fraction"
FOCUSED_PANELS = [c for c in CONTRASTS if c[0] != "CD8A vs Control"]
FOCUSED_METRICS = [("factin_fraction", "F-actin fraction of cell area", "factin"),
                   ("fa_fraction", "Focal adhesion fraction of cell area", "focal_adhesion"),
                   ("microvilli_fraction", "Microvilli fraction of cell area", "microvilli")]
METRIC_LABELS = {"dice": "Dice", "cldice": "clDice", "precision": "Precision", "recall": "Recall", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}


def label(group: str, one_line: bool = False) -> str:
    """Display name of a group."""
    text = LABELS.get(group, group)
    return text.replace("\n", " ") if one_line else text


def save(fig, out_dir: str, name: str) -> None:
    """Save a figure as 300 dpi PNG and vector PDF."""
    os.makedirs(os.path.dirname(os.path.join(out_dir, name)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def example_images(val_df: pd.DataFrame, groups: list, mode: str) -> dict:
    """Per group, the test image with the highest microvilli Dice ("best") or the one closest to the group median ("median")."""
    picks = {}
    for g in groups:
        sub = val_df[(val_df["group"] == g) & (val_df["class"] == "microvilli")].dropna(subset=["dice"])
        if len(sub):
            index = sub["dice"].idxmax() if mode == "best" else (sub["dice"] - sub["dice"].median()).abs().idxmin()
            picks[g] = sub.loc[index]
    return picks


def square_crop(mask: np.ndarray, margin: float = 0.15) -> tuple:
    """Square (top, bottom, left, right) around the labeled cell with a margin, clipped to the image."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0, mask.shape[0], 0, mask.shape[1]
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) * (0.5 + margin)
    half = min(half, min(mask.shape) / 2)
    cy = min(max(cy, half), mask.shape[0] - half)
    cx = min(max(cx, half), mask.shape[1] - half)
    return int(cy - half), int(cy + half), int(cx - half), int(cx + half)


def colorize(mask: np.ndarray) -> np.ndarray:
    """RGB image of a class-id mask using the thesis mask colors."""
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for class_id, hex_color in MASK_COLORS.items():
        rgb[mask == class_id] = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    return rgb


def figure1(picks: dict, settings: dict, model, device, out_dir: str) -> None:
    """3 x N grid: raw image, manual mask and model mask (rows) for one representative test cell per group (columns)."""
    groups = list(picks)
    fig, axes = plt.subplots(3, len(groups), figsize=(1.25 * len(groups) + 0.6, 4.6))
    for col, g in enumerate(groups):
        name = picks[g]["image"]
        image_path = train_TIRF.resolve_image_path(settings["image_dir"], name)
        image = np.array(Image.open(image_path).convert("L")).astype(float)
        gt = np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L"))
        pred = predict_mask(model, image_path, device, settings["class_thresholds"], settings["threshold_fallback"])
        t, b, l, r = square_crop(gt)
        crop = image[t:b, l:r]
        axes[0, col].imshow(crop, cmap="gray", vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99.5))
        axes[1, col].imshow(colorize(gt[t:b, l:r]))
        axes[2, col].imshow(colorize(pred[t:b, l:r]))
        axes[0, col].set_title(label(g), fontsize=8, color=INK)
        axes[2, col].text(0.5, -0.06, f"Dice {picks[g]['dice']:.2f}", transform=axes[2, col].transAxes,
                          ha="center", va="top", fontsize=7, color=INK_SECONDARY)
        for row in range(3):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
                spine.set_linewidth(0.6)
    for row, text in enumerate(["Image", "Manual", "Model"]):
        axes[row, 0].set_ylabel(text, fontsize=9, color=INK)
    fig.legend(handles=[Patch(facecolor=MASK_COLORS[c], edgecolor="#c3c2b7", label=t) for c, t in MASK_LEGEND],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save(fig, out_dir, "evaluation/examples")


def figure2(val_df: pd.DataFrame, groups: list, out_dir: str) -> None:
    """Per-structure accuracy on held-out test images: F-actin clDice, focal adhesion Dice and cell body Dice, one box per group."""
    panels = [("F-actin", "cldice", "F-actin: clDice"), ("Focal Adhesion", "dice", "Focal adhesion: Dice"),
              ("Lamellipodia", "dice", "Cell body (lamellipodia): Dice")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 8.4), sharex=True)
    for k, (ax, (cls, metric, title)) in enumerate(zip(axes, panels)):
        sub = val_df[val_df["class"] == cls]
        box_with_dots(ax, [sub.loc[sub["group"] == g, metric] for g in groups], MODEL_COLOR, np.random.default_rng(k))
        for i, g in enumerate(groups):
            missing = int(sub.loc[sub["group"] == g, metric].isna().sum())
            if missing:
                ax.text(i, 0.03, f"{missing} undefined", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        if "CD8AA_old" in groups:
            ax.axvline(groups.index("CD8AA_old") - 0.5, color="#c3c2b7", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(METRIC_LABELS[metric], color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style_axis(ax)
    axes[-1].set_xticks(range(len(groups)))
    axes[-1].set_xticklabels([f"{label(g)}\nn = {(sub['group'] == g).sum()}" for g in groups], fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "evaluation/segmentation_accuracy")


def concordance(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    return 2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2)


def figure3(fraction_df: pd.DataFrame, groups: list, out_dir: str) -> pd.DataFrame:
    """Bland-Altman plot of model vs manual microvilli area fraction, one panel per group."""
    ncols = 3
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.6 * nrows + 0.4), sharex=True, sharey=True, squeeze=False)
    d = fraction_df.dropna(subset=["manual_fraction", "predicted_fraction"])
    mean_all = (d["manual_fraction"] + d["predicted_fraction"]) / 2
    diff_all = d["predicted_fraction"] - d["manual_fraction"]
    rows = []
    for ax, g in zip(axes.flat, groups):
        s = d[d["group"] == g]
        manual, model = s["manual_fraction"].to_numpy(), s["predicted_fraction"].to_numpy()
        mean, diff = (manual + model) / 2, model - manual
        bias, sd = diff.mean(), diff.std(ddof=1)
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        ccc = concordance(manual, model)
        rows.append({"group": g, "n": len(s), "bias": bias, "loa_low": lo, "loa_high": hi, "ccc": ccc})
        ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        ax.axhline(bias, color=INK, linewidth=1.2)
        for y in (lo, hi):
            ax.axhline(y, color=INK_SECONDARY, linewidth=0.9, linestyle="--")
        ax.scatter(mean, diff, s=14, color=MODEL_COLOR, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(label(g, one_line=True), fontsize=9, color=INK)
        ax.text(0.03, 0.97, f"n = {len(s)}\nbias = {bias:+.3f}\nLoA {lo:+.3f} to {hi:+.3f}\nCCC = {ccc:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color=INK)
        style_axis(ax)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    pad_y = 0.08 * (diff_all.max() - diff_all.min())
    axes[0, 0].set_ylim(diff_all.min() - pad_y, diff_all.max() + pad_y * 3)
    axes[0, 0].set_xlim(0, mean_all.max() * 1.05)
    fig.supxlabel("Mean of manual and model microvilli fraction (microvilli area / cell area)", fontsize=9, color=INK)
    fig.supylabel("Model − manual microvilli fraction", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out_dir, "evaluation/bland_altman")
    return pd.DataFrame(rows)


def table1(val_df: pd.DataFrame, groups: list, out_dir: str) -> pd.DataFrame:
    """Mean ± SD of every segmentation metric per group and class on held-out test images."""
    rows = []
    for g in groups:
        for c in val_df["class"].cat.categories if hasattr(val_df["class"], "cat") else val_df["class"].unique():
            s = val_df[(val_df["group"] == g) & (val_df["class"] == c)]
            if not len(s):
                continue
            row = {"Group": label(g, one_line=True), "Class": c, "n": s["image"].nunique()}
            for metric, title in METRIC_LABELS.items():
                v = s[metric].dropna()
                row[title] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "–")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "evaluation", "segmentation_metrics.csv"), index=False, encoding="utf-8-sig")
    return table


def class_score_rows(val_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD over test images of Dice, clDice, precision, recall and F1 (pixel F1 equals Dice) for each class."""
    rows = []
    for c in dict.fromkeys(val_df["class"]):
        s = val_df[val_df["class"] == c]
        row = {"Class": CLASS_LABELS.get(c, c), "n images": s["image"].nunique()}
        for metric, title in (("dice", "Dice"), ("cldice", "clDice"), ("precision", "Precision"), ("recall", "Recall"), ("dice", "F1")):
            v = s[metric].dropna()
            row[title] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "–")
        rows.append(row)
    return pd.DataFrame(rows)


def class_score_tables(val_df: pd.DataFrame, groups: list, out_dir: str) -> pd.DataFrame:
    """Per-class scores pooled over all phenotypes (CSV), and one Excel sheet per phenotype."""
    overall = class_score_rows(val_df)
    overall.to_csv(os.path.join(out_dir, "evaluation", "class_scores.csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(out_dir, "evaluation", "class_scores_by_phenotype.xlsx")) as writer:
        overall.to_excel(writer, sheet_name="All phenotypes", index=False)
        for g in groups:
            class_score_rows(val_df[val_df["group"] == g]).to_excel(writer, sheet_name=label(g, one_line=True), index=False)
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = max(len(str(cell.value)) for cell in column) + 2
    return overall


def table2(run_dir: str, groups: list, out_dir: str) -> pd.DataFrame:
    """Number of training (train+val pool) and held-out test images per group, from the run's split.json."""
    with open(os.path.join(run_dir, "split.json")) as f:
        split = json.load(f)
    group_of = split["groups"]
    test = set(split["test"])
    rows = []
    for g in groups:
        names = [n for n, grp in group_of.items() if grp == g]
        n_test = sum(n in test for n in names)
        rows.append({"Group": label(g, one_line=True), "Training (train + validation)": len(names) - n_test,
                     "Test (held out)": n_test, "Total": len(names)})
    table = pd.DataFrame(rows)
    table.loc[len(table)] = ["Total"] + table.iloc[:, 1:].sum().tolist()
    table.to_csv(os.path.join(out_dir, "evaluation", "dataset_split.csv"), index=False, encoding="utf-8-sig")
    return table


def metric_values(features: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-image manual and model values of one feature, as manual_value / predicted_value."""
    return features[["image", "group", "predicted_by", f"manual_{metric}", f"predicted_{metric}"]].rename(
        columns={f"manual_{metric}": "manual_value", f"predicted_{metric}": "predicted_value"})


def typical_cells(values_df: pd.DataFrame, groups: list) -> dict:
    """Per group, the image whose manual value of the compared measure is closest to the group median."""
    picks = {}
    for g in groups:
        sub = values_df[values_df["group"] == g].dropna(subset=["manual_value"])
        picks[g] = sub.loc[(sub["manual_value"] - sub["manual_value"].median()).abs().idxmin(), "image"]
    return picks


def show_thumbnail(ax, name: str, settings: dict) -> None:
    """Raw image cropped around its labeled cell, without axes."""
    image = np.array(Image.open(train_TIRF.resolve_image_path(settings["image_dir"], name)).convert("L")).astype(float)
    gt = np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L"))
    t, b, l, r = square_crop(gt)
    crop = image[t:b, l:r]
    ax.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
        spine.set_linewidth(0.6)


def paired_boxes(ax, values_df: pd.DataFrame, groups: list, rng) -> None:
    """Manual and model boxes side by side at each group position, every cell drawn as a dot."""
    for offset, (source, _, color) in zip((-0.2, 0.2), SOURCES):
        for i, g in enumerate(groups):
            data = values_df.loc[values_df["group"] == g, f"{source}_value"].dropna().to_numpy()
            box = ax.boxplot([data], positions=[i + offset], widths=0.34, patch_artist=True, showfliers=False,
                             medianprops={"color": INK, "linewidth": 1.4}, whiskerprops={"color": INK_SECONDARY},
                             capprops={"color": INK_SECONDARY}, boxprops={"edgecolor": INK_SECONDARY})
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(0.35)
            ax.scatter(i + offset + rng.uniform(-0.08, 0.08, data.size), data, s=9, color=color,
                       edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{label(g)}\nn = {(values_df['group'] == g).sum()}" for g in groups], fontsize=8)
    style_axis(ax)


def source_legend(fig, **kwargs) -> None:
    """Manual / Model legend."""
    fig.legend(handles=[Patch(facecolor=c, alpha=0.6, edgecolor=INK_SECONDARY, label=t) for _, t, c in SOURCES],
               frameon=False, fontsize=8, **kwargs)


def contrast_stats(values_df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Planned contrasts per source on the compared measure: median difference (B - A) with bootstrap 95% CI, Cliff's delta, Mann-Whitney p, Holm-adjusted."""
    rng = np.random.default_rng(seed)
    rows = []
    for source, source_name, _ in SOURCES:
        block = []
        for family, a, b in CONTRASTS:
            x = values_df.loc[values_df["group"] == a, f"{source}_value"].dropna().to_numpy()
            y = values_df.loc[values_df["group"] == b, f"{source}_value"].dropna().to_numpy()
            boot = np.median(rng.choice(y, (n_boot, y.size)), axis=1) - np.median(rng.choice(x, (n_boot, x.size)), axis=1)
            u, p = stats.mannwhitneyu(y, x, alternative="two-sided")
            block.append({"source": source_name, "family": family, "comparison": f"{label(b, True)} vs {label(a, True)}",
                          "group_a": a, "group_b": b, "n_a": x.size, "n_b": y.size,
                          "median_a": np.median(x), "median_b": np.median(y), "median_diff": np.median(y) - np.median(x),
                          "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                          "cliffs_delta": 2 * u / (x.size * y.size) - 1, "p_value": p})
        for row, adjusted in zip(block, holm([r["p_value"] for r in block])):
            row["p_holm"] = adjusted
        rows += block
    return pd.DataFrame(rows)


def p_text(p: float) -> str:
    """Compact p-value label."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def figure4(values_df: pd.DataFrame, groups: list, settings: dict, out_dir: str) -> None:
    """Compared measure per group, manual vs model, with a typical cell above each group."""
    picks = typical_cells(values_df, groups)
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, len(groups), height_ratios=[1, 2.6], hspace=0.08, wspace=0.08)
    for i, g in enumerate(groups):
        show_thumbnail(fig.add_subplot(grid[0, i]), picks[g], settings)
    ax = fig.add_subplot(grid[1, :])
    paired_boxes(ax, values_df, groups, np.random.default_rng(0))
    ax.set_ylim(bottom=0)
    ax.set_ylabel(COMPARE_LABEL, fontsize=9, color=INK)
    source_legend(fig, loc="upper right", bbox_to_anchor=(0.9, 0.62), ncol=2)
    save(fig, out_dir, "comparison/factin_overview")


def figure5(values_df: pd.DataFrame, contrasts: pd.DataFrame, picks: dict, ylabel: str, name: str, settings: dict, out_dir: str) -> None:
    """Focused pairwise comparisons (Full vs Trunc, GFP vs Ab), each with typical-cell thumbnails and manual/model values."""
    panels = FOCUSED_PANELS
    top = np.nanmax(values_df.loc[values_df["group"].isin(list(picks)), ["manual_value", "predicted_value"]].to_numpy())
    fig = plt.figure(figsize=(7.2, 8.4))
    outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, top=0.95, bottom=0.14)
    for k, (family, a, b) in enumerate(panels):
        cell = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[k // 2, k % 2], height_ratios=[1, 1.9], hspace=0.06, wspace=0.06)
        thumbs = [fig.add_subplot(cell[0, j]) for j in range(2)]
        for ax_t, g in zip(thumbs, (a, b)):
            show_thumbnail(ax_t, picks[g], settings)
        thumbs[0].text(1.03, 1.08, f"{family.split(' (')[0]}: {label(b, True)} vs {label(a, True)}", transform=thumbs[0].transAxes,
                       ha="center", va="bottom", fontsize=9, color=INK)
        ax = fig.add_subplot(cell[1, :])
        paired_boxes(ax, values_df, [a, b], np.random.default_rng(k))
        ax.set_ylim(0, top * 1.14)
        if k % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=INK)
        res = contrasts[(contrasts["group_a"] == a) & (contrasts["group_b"] == b)].set_index("source")
        ax.text(0.5, 0.98, f"Manual {p_text(res.loc['Manual', 'p_holm'])}    Model {p_text(res.loc['Model', 'p_holm'])}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK)
    source_legend(fig, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.01, "p: two-sided Mann-Whitney U, Holm-adjusted over the 8 planned comparisons",
             ha="center", fontsize=7, color=INK_SECONDARY)
    save(fig, out_dir, name)


def figure6(contrasts: pd.DataFrame, out_dir: str) -> None:
    """Forest plot: median difference with bootstrap 95% CI for each planned comparison, manual vs model."""
    y_positions, ticks, y = {}, [], 0.0
    for fam in dict.fromkeys(contrasts["family"]):
        y -= 0.7
        ticks.append((y, fam, True))
        for comp in contrasts.loc[contrasts["family"] == fam, "comparison"].drop_duplicates():
            y -= 1
            y_positions[comp] = y
            ticks.append((y, comp, False))
    right = contrasts["ci_high"].max()
    span = right - min(contrasts["ci_low"].min(), 0)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(y_positions) + 1.8))
    for offset, (_, name, color) in zip((0.17, -0.17), SOURCES):
        res = contrasts[contrasts["source"] == name]
        ys = np.array([y_positions[c] for c in res["comparison"]]) + offset
        ax.errorbar(res["median_diff"], ys, xerr=[res["median_diff"] - res["ci_low"], res["ci_high"] - res["median_diff"]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=2.5, markersize=5, label=name)
        for yy, p in zip(ys, res["p_holm"]):
            ax.text(right + 0.04 * span, yy, p_text(p), va="center", fontsize=7, color=color)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.9)
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=8)
    for tick, (_, _, header) in zip(ax.get_yticklabels(), ticks):
        tick.set_fontweight("bold" if header else "normal")
    ax.set_ylim(y - 0.8, 0)
    ax.set_xlim(min(contrasts["ci_low"].min(), 0) - 0.06 * span, right + 0.3 * span)
    ax.set_xlabel(f"Difference in median {COMPARE_SHORT} (first group minus second)",
                  fontsize=8, color=INK)
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, out_dir, "comparison/effect_sizes")


def table3(contrasts: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Planned-comparison statistics, manual and model side by side."""
    rows = []
    for comp in dict.fromkeys(contrasts["comparison"]):
        res = contrasts[contrasts["comparison"] == comp].set_index("source")
        row = {"Comparison": comp, "Question": res["family"].iloc[0], "n": f"{res['n_b'].iloc[0]} vs {res['n_a'].iloc[0]}"}
        for source in ("Manual", "Model"):
            r = res.loc[source]
            row[f"{source} median difference [95% CI]"] = f"{r['median_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            row[f"{source} Cliff's delta"] = f"{r['cliffs_delta']:+.2f}"
            row[f"{source} p (Holm)"] = f"{r['p_holm']:.2g}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "comparison", "group_comparisons.csv"), index=False, encoding="utf-8-sig")
    full = contrasts.round(4)
    full[["group_a", "group_b"]] = full[["group_a", "group_b"]].map(lambda g: label(g, one_line=True))
    full.to_csv(os.path.join(out_dir, "comparison", "group_comparisons_full.csv"), index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder. Default: config's out-dir.")
    parser.add_argument("--quant-dir", default=None, help="quantification.py output. Default: <run-dir>/quantification")
    parser.add_argument("--out-dir", default="thesis_figures")
    parser.add_argument("--examples", choices=["best", "median"], default="best",
                        help="Figure 1 example per group: highest microvilli Dice, or closest to the group median.")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    quant_dir = args.quant_dir or os.path.join(settings["run_dir"], "quantification")
    for sub in ("evaluation", "comparison"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    val_df = pd.read_csv(os.path.join(quant_dir, "validation_per_image.csv"))
    fraction_df = pd.read_csv(os.path.join(quant_dir, "fraction_per_image.csv"))
    features_path = os.path.join(quant_dir, "features_per_image.csv")
    if not os.path.exists(features_path):
        raise SystemExit(f"{features_path} not found: run scripts/extract_features.sh before thesis_figures.sh.")
    features = pd.read_csv(features_path)
    values_df = metric_values(features, COMPARE_METRIC)
    val_df["class"] = pd.Categorical(val_df["class"], categories=list(dict.fromkeys(val_df["class"])), ordered=True)
    groups = [g for g in settings["groups"] if (val_df["group"] == g).any()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(settings["run_dir"], settings["classes"], device)
    picks = example_images(val_df, groups, args.examples)
    figure1(picks, settings, model, device, args.out_dir)
    figure2(val_df, groups, args.out_dir)
    agreement = figure3(fraction_df, groups, args.out_dir)
    agreement.round(3).to_csv(os.path.join(args.out_dir, "evaluation", "bland_altman_agreement.csv"), index=False)
    table = table1(val_df, groups, args.out_dir)
    class_scores = class_score_tables(val_df, groups, args.out_dir)
    split_table = table2(settings["run_dir"], groups, args.out_dir)
    compare = [g for g in COMPARISON_ORDER if g in settings["compare_groups"]]
    contrasts = contrast_stats(values_df)
    figure4(values_df, compare, settings, args.out_dir)
    focused_picks = typical_cells(values_df, sorted({g for _, a, b in FOCUSED_PANELS for g in (a, b)}))
    focused = []
    for metric, ylabel, suffix in FOCUSED_METRICS:
        metric_df = metric_values(features, metric)
        metric_contrasts = contrasts if metric == COMPARE_METRIC else contrast_stats(metric_df)
        figure5(metric_df, metric_contrasts, focused_picks, ylabel, f"comparison/focused_comparisons_{suffix}", settings, args.out_dir)
        focused.append(metric_contrasts.assign(metric=metric))
    focused = pd.concat(focused)
    focused = focused[focused["family"] != "CD8A vs Control"].round(4)
    focused[["group_a", "group_b"]] = focused[["group_a", "group_b"]].map(lambda g: label(g, one_line=True))
    focused[["metric"] + [c for c in focused.columns if c != "metric"]].to_csv(
        os.path.join(args.out_dir, "comparison", "focused_comparisons_stats.csv"), index=False)
    figure6(contrasts, args.out_dir)
    comparison_table = table3(contrasts, args.out_dir)

    print(f"Figure 1 examples ({args.examples} microvilli Dice per group, test images):")
    for g, r in picks.items():
        print(f"  {label(g, one_line=True):22s} {r['image']}  Dice {r['dice']:.3f}")
    print("\nFigure 3 agreement (model - manual microvilli fraction):")
    print(agreement.round(2).to_string(index=False))
    print(f"\nTable 3: planned comparisons ({COMPARE_SHORT} difference = first listed group minus second)")
    print(comparison_table.to_string(index=False))
    print("\nTable 2: images per group")
    print(split_table.to_string(index=False))
    print(f"\nTable 1: {len(table)} rows. Everything saved to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
