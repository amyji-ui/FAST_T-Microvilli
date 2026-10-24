#!/usr/bin/env python
"""Validate the model against manual labels per group, and compare microvilli density between groups."""

import argparse
import glob
import json
import os
import shutil
from itertools import combinations
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from skimage.morphology import skeletonize
from torchvision import transforms

import train_TIRF

MANUAL_COLOR = "#8a8986"
MODEL_COLOR = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
DENSITY_UNIT = 10000


def load_settings(config_path: str, run_dir: Optional[str]) -> dict:
    """Resolve training settings plus quantification-only keys from config.yaml."""
    train_args = train_TIRF.build_arg_parser().parse_args(["--config", config_path])
    settings = train_TIRF.load_config(train_args)
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    settings["project_dir"] = raw.get("project-dir")
    class_names = {0: "Background"}
    for name, class_id in raw["class-name-to-index"].items():
        class_names.setdefault(int(class_id), name)
    if sorted(class_names) != list(range(settings["classes"])):
        raise ValueError("class-name-to-index must define class ids 1..classes-1 (background is 0)")

    settings["class_names"] = class_names
    settings["groups"] = raw.get("groups") or {}
    settings["count_min_area"] = raw.get("count-min-area") or {}

    compare = raw.get("compare-groups")
    compare = list(settings["groups"]) if compare is None else list(compare)
    unknown = [g for g in compare if g not in settings["groups"]]
    if unknown:
        raise ValueError(f"compare-groups lists unknown group(s) {unknown}; groups are {list(settings['groups'])}")
    settings["compare_groups"] = compare

    name_to_id = {name: cid for cid, name in class_names.items()}
    merged_classes = {}
    for merged_name, member_names in (raw.get("merged-classes") or {}).items():
        unknown = [n for n in member_names if n not in name_to_id]
        if unknown:
            raise ValueError(f"merged-classes '{merged_name}' lists unknown class(es) {unknown}; known classes are {list(name_to_id)}")
        merged_classes[merged_name] = [name_to_id[n] for n in member_names]
    settings["merged_classes"] = merged_classes

    raw_thresholds = raw.get("class-thresholds") or {}
    unknown = [n for n in raw_thresholds if n not in name_to_id]
    if unknown:
        raise ValueError(f"class-thresholds lists unknown class(es) {unknown}; known classes are {list(name_to_id)}")
    settings["class_thresholds"] = {name_to_id[n]: float(v) for n, v in raw_thresholds.items()}
    settings["threshold_fallback"] = raw.get("threshold-fallback", "next_best")

    settings["run_dir"] = run_dir or settings["out_dir"]
    return settings


assign_group = train_TIRF.assign_group


def load_fold_models(run_dir: str, classes: int, device: torch.device) -> list:
    """Load every k-fold checkpoint in run_dir/model, in fold order."""
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
    return members


def load_model(run_dir: str, classes: int, device: torch.device) -> torch.nn.Module:
    """Load every k-fold checkpoint in run_dir/model as one averaging ensemble."""
    ensemble = train_TIRF.EnsembleModel(load_fold_models(run_dir, classes, device)).to(device)
    ensemble.eval()
    return ensemble


@torch.no_grad()
def predict_probs(model: torch.nn.Module, image_path: str, device: torch.device) -> torch.Tensor:
    """Softmax class probabilities (C, H, W) for one image."""
    image = Image.open(image_path).convert("L")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return model(tensor)[0]


def probs_to_mask(probs: torch.Tensor, class_thresholds: Optional[dict] = None, threshold_fallback: str = "next_best") -> np.ndarray:
    """Class-id mask from probabilities: argmax, with optional per-class confidence thresholds."""
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


def predict_mask(model: torch.nn.Module, image_path: str, device: torch.device,
                 class_thresholds: Optional[dict] = None, threshold_fallback: str = "next_best") -> np.ndarray:
    """Predict the (H, W) class-id mask for one image."""
    return probs_to_mask(predict_probs(model, image_path, device), class_thresholds, threshold_fallback)


def cl_dice(g: np.ndarray, p: np.ndarray) -> float:
    """clDice (Shit et al., 2021): skeleton-to-region overlap between two boolean masks."""
    skel_g, skel_p = skeletonize(g), skeletonize(p)
    if not skel_g.any() or not skel_p.any():
        return float("nan")
    t_prec = (skel_p & g).sum() / skel_p.sum()
    t_sens = (skel_g & p).sum() / skel_g.sum()
    return 2 * t_prec * t_sens / (t_prec + t_sens) if (t_prec + t_sens) else float("nan")


def pixel_metrics_bool(g: np.ndarray, p: np.ndarray, cldice: bool = True) -> dict:
    """Dice, clDice, precision and recall between a manual and a predicted boolean mask."""
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    nan = float("nan")
    return {
        "dice": 2 * tp / (g.sum() + p.sum()) if (g.sum() + p.sum()) else nan,
        "cldice": cl_dice(g, p) if cldice else nan,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
    }


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Pixel-level ROC-AUC via the Mann-Whitney rank formula (ties averaged)."""
    y = y.ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = stats.rankdata(score.ravel())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Pixel-level PR-AUC as average precision (sum of precision x recall steps, ties grouped)."""
    y = y.ravel().astype(bool)
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    order = np.argsort(-score.ravel(), kind="mergesort")
    s, t = score.ravel()[order], y[order]
    last_of_tie = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(t)[last_of_tie]
    precision = tp / (last_of_tie + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def count_objects_bool(bool_mask: np.ndarray, min_area: float) -> int:
    """Number of outer contours in a boolean mask with area >= min_area."""
    contours, _ = cv2.findContours(bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def seg_classes(settings: dict) -> dict:
    """{display name: member class ids} for every foreground class plus each merged class."""
    out = {name: [cid] for cid, name in settings["class_names"].items() if cid != 0}
    out.update(settings["merged_classes"])
    return out


def validation_rows(name: str, group: str, gt: np.ndarray, probs: np.ndarray, pred: np.ndarray, settings: dict) -> list:
    """Per-class Dice, clDice, precision, recall, ROC-AUC and PR-AUC for one test image."""
    cldice_ids = set(settings["cldice_class_ids"])
    rows = []
    for class_name, member_ids in seg_classes(settings).items():
        g, p = np.isin(gt, member_ids), np.isin(pred, member_ids)
        score = probs[member_ids].sum(axis=0)
        filamentous = bool(cldice_ids & set(member_ids))
        rows.append({"image": name, "group": group, "class": class_name,
                     **pixel_metrics_bool(g, p, cldice=filamentous),
                     "roc_auc": roc_auc(g, score), "pr_auc": average_precision(g, score)})
    return rows


def density_row(name: str, group: str, gt: np.ndarray, pred: np.ndarray, predicted_by: str, settings: dict) -> dict:
    """Microvilli count, cell footprint (holes filled), density and area fraction for one image, manual and predicted."""
    member_ids = settings["merged_classes"]["microvilli"]
    min_area = settings["count_min_area"].get("microvilli", 0)
    row = {"image": name, "group": group, "predicted_by": predicted_by}
    for source, mask in (("manual", gt), ("predicted", pred)):
        cell = int(ndi.binary_fill_holes(mask != 0).sum())
        microvilli = np.isin(mask, member_ids)
        count = count_objects_bool(microvilli, min_area)
        row[f"{source}_count"] = count
        row[f"{source}_cell_px"] = cell
        row[f"{source}_density"] = count / cell * DENSITY_UNIT if cell else float("nan")
        row[f"{source}_area_fraction"] = microvilli.sum() / cell if cell else float("nan")
    return row


def holm(p_values: list) -> list:
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def compare_density(density_df: pd.DataFrame, groups: list) -> pd.DataFrame:
    """Kruskal-Wallis across groups, then pairwise Mann-Whitney U with Holm correction, for manual and predicted density."""
    rows = []
    for source in ("manual", "predicted"):
        samples = {g: density_df.loc[density_df["group"] == g, f"{source}_density"].dropna().to_numpy() for g in groups}
        samples = {g: v for g, v in samples.items() if len(v) >= 2}
        if len(samples) >= 3:
            h, p = stats.kruskal(*samples.values())
            rows.append({"source": source, "comparison": "all groups", "test": "Kruskal-Wallis", "n_a": sum(map(len, samples.values())),
                         "n_b": None, "median_a": None, "median_b": None, "statistic": h, "p_value": p, "p_holm": p})
        pairs = []
        for a, b in combinations(samples, 2):
            u, p = stats.mannwhitneyu(samples[a], samples[b], alternative="two-sided")
            pairs.append({"source": source, "comparison": f"{a} vs {b}", "test": "Mann-Whitney U", "n_a": len(samples[a]),
                          "n_b": len(samples[b]), "median_a": np.median(samples[a]), "median_b": np.median(samples[b]),
                          "statistic": u, "p_value": p})
        for pair, adjusted in zip(pairs, holm([r["p_value"] for r in pairs])):
            pair["p_holm"] = adjusted
        rows.extend(pairs)
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    """Significance label for a Holm-adjusted p-value."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def style_axis(ax) -> None:
    """Recessive grid and spines."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)


def box_with_dots(ax, data_by_x: list, color: str, rng) -> None:
    """One box per x position with every image drawn as a dot."""
    for i, data in enumerate(data_by_x):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            ax.text(i, 0.5, "n/a", ha="center", va="center", color=INK_SECONDARY, fontsize=9, transform=ax.get_xaxis_transform())
            continue
        box = ax.boxplot([data], positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_SECONDARY},
                         capprops={"color": INK_SECONDARY}, boxprops={"edgecolor": INK_SECONDARY})
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.35)
        ax.scatter(i + rng.uniform(-0.14, 0.14, data.size), data, s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3)


VALIDATION_METRICS = {"dice": "Dice", "cldice": "clDice", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def plot_validation(val_df: pd.DataFrame, group: str, metric: str, class_order: list, path: str) -> None:
    """One metric per segmentation class for one group's test images."""
    sub = val_df[val_df["group"] == group]
    n = sub["image"].nunique()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    box_with_dots(ax, [sub.loc[sub["class"] == c, metric] for c in class_order], MODEL_COLOR, np.random.default_rng(0))
    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in class_order], fontsize=10)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel(f"{VALIDATION_METRICS[metric]} (1 = perfect)", color=INK)
    ax.set_title(f"{group}: {VALIDATION_METRICS[metric]}, model vs manual\n{n} held-out test image{'s' if n != 1 else ''} (dot = image)",
                 color=INK, fontsize=11)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_density(density_df: pd.DataFrame, comparison: pd.DataFrame, groups: list, source: str, top: float, path: str) -> None:
    """Microvilli density per group from one source, with Holm-adjusted significance brackets."""
    title, color = {"manual": ("Manual labels", MANUAL_COLOR), "predicted": ("Model, out-of-fold predictions", MODEL_COLOR)}[source]
    fig, ax = plt.subplots(figsize=(1.5 * len(groups) + 1.5, 5.4))
    box_with_dots(ax, [density_df.loc[density_df["group"] == g, f"{source}_density"] for g in groups], color, np.random.default_rng(0))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("_", "\n", 1) for g in groups], fontsize=10)
    style_axis(ax)
    sub = comparison[comparison["source"] == source]
    kw = sub[sub["test"] == "Kruskal-Wallis"]
    kw_text = f"Kruskal-Wallis p = {kw['p_value'].iloc[0]:.2g}" if len(kw) else ""
    ax.set_title(f"Microvilli density: {title}\n{kw_text}", color=INK, fontsize=11)
    level = 0
    for _, r in sub[(sub["test"] == "Mann-Whitney U") & (sub["p_holm"] < 0.05)].iterrows():
        a, b = (groups.index(x) for x in r["comparison"].split(" vs "))
        y = top * (1.06 + 0.08 * level)
        ax.plot([a, a, b, b], [y - top * 0.015, y, y, y - top * 0.015], color=INK_SECONDARY, linewidth=1)
        ax.text((a + b) / 2, y, stars(r["p_holm"]), ha="center", va="bottom", color=INK, fontsize=9)
        level += 1
    ax.set_ylim(0, top * (1.14 + 0.08 * max(level, 1)))
    ax.set_ylabel(f"microvilli per {DENSITY_UNIT:,} px of cell (dot = cell)", color=INK)
    fig.text(0.5, 0.005, "Brackets: pairwise Mann-Whitney U, Holm-adjusted (* p<0.05, ** p<0.01, *** p<0.001)",
             ha="center", va="bottom", color=INK_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_overlay(name, image_dir, gt, pred, counts, colormap, path) -> None:
    """Image, manual mask and predicted mask side by side for one test image, with microvilli counts."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    image = np.array(Image.open(train_TIRF.resolve_image_path(image_dir, name)).convert("L"))
    axes[0].imshow(image, cmap="gray", vmax=np.percentile(image, 99.5))
    axes[0].set_title(name.split("/", 1)[1], fontsize=10)
    axes[1].imshow(colormap[gt])
    axes[1].set_title(f"Manual: {counts['manual_count']} microvilli", fontsize=10)
    axes[2].imshow(colormap[pred])
    axes[2].set_title(f"Model: {counts['predicted_count']} microvilli", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fold_of_images(split: dict) -> tuple:
    """({image: fold index whose model never trained on it}, set of test images) from split.json."""
    val_fold = {n: i for i, fold in enumerate(split.get("folds", [])) for n in fold["val"]}
    return val_fold, set(split.get("test", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="Training output folder (model/fold*_model.pth + split.json). Default: config's out-dir.")
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/quantification")
    args = parser.parse_args()

    settings = load_settings(args.config, args.run_dir)
    if "microvilli" not in settings["merged_classes"]:
        raise SystemExit("config.yaml merged-classes must define 'microvilli' (the structure whose density is compared).")
    run_dir = settings["run_dir"]
    out_dir = args.out_dir or os.path.join(run_dir, "quantification")
    os.makedirs(out_dir, exist_ok=True)

    split_path = os.path.join(run_dir, "split.json")
    split = json.load(open(split_path)) if os.path.exists(split_path) else {}
    val_fold, split_test = fold_of_images(split)
    if not val_fold:
        print(f"WARNING: no fold assignments in {split_path} -- every image is predicted by the full ensemble (optimistic for training images).")

    names = train_TIRF.list_dataset_images(settings["image_dir"])
    test_names = train_TIRF.test_split(names, settings["image_dir"])
    if set(test_names) != split_test and split_test:
        print("WARNING: the current _Test datasets differ from the test set this model was trained with (split.json).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = load_fold_models(run_dir, settings["classes"], device)
    ensemble = train_TIRF.EnsembleModel(folds).to(device).eval()
    fold_models = [train_TIRF.EnsembleModel([m]).to(device).eval() for m in folds]
    thresholds, fallback = settings["class_thresholds"], settings["threshold_fallback"]
    compare = settings["compare_groups"]
    print(f"Model: {run_dir} ({len(folds)} fold models) on {device}")

    val_rows, density_rows, gts, preds = [], [], {}, {}
    for name in names:
        group = assign_group(name, settings["groups"])
        is_test = name in test_names
        if not is_test and group not in compare:
            continue
        if is_test or name not in val_fold:
            model, predicted_by = ensemble, "ensemble (test)" if is_test else "ensemble (not in split)"
        else:
            model, predicted_by = fold_models[val_fold[name]], f"fold {val_fold[name]} (out-of-fold)"
        probs = predict_probs(model, train_TIRF.resolve_image_path(settings["image_dir"], name), device)
        pred = probs_to_mask(probs, thresholds, fallback)
        gt = np.array(Image.open(train_TIRF.resolve_mask_path(settings["mask_dir"], name)).convert("L"))
        if is_test:
            val_rows += validation_rows(name, group, gt, probs.cpu().numpy(), pred, settings)
            gts[name], preds[name] = gt, pred
        if group in compare:
            density_rows.append(density_row(name, group, gt, pred, predicted_by, settings))

    class_order = list(seg_classes(settings))
    group_order = [g for g in list(settings["groups"]) + ["Unassigned"]]
    val_df = pd.DataFrame(val_rows)
    val_df["group"] = pd.Categorical(val_df["group"], categories=group_order, ordered=True)
    val_df["class"] = pd.Categorical(val_df["class"], categories=class_order, ordered=True)
    val_df = val_df.sort_values(["group", "class", "image"])
    density_df = pd.DataFrame(density_rows)
    density_df["group"] = pd.Categorical(density_df["group"], categories=compare, ordered=True)
    density_df = density_df.sort_values(["group", "image"])

    metrics = ["dice", "cldice", "precision", "recall", "roc_auc", "pr_auc"]
    grouped = val_df.groupby(["group", "class"], observed=True)
    val_summary = grouped[metrics].agg(["mean", "std"])
    val_summary.columns = [f"{m}_{s}" for m, s in val_summary.columns]
    val_summary.insert(0, "n_images", grouped["image"].nunique())
    val_summary = val_summary.reset_index()
    comparison = compare_density(density_df, compare)
    density_summary = density_df.groupby("group", observed=True)[
        ["manual_density", "predicted_density", "manual_area_fraction", "predicted_area_fraction", "manual_cell_px"]
    ].median().round(4)
    density_summary.insert(0, "n_images", density_df.groupby("group", observed=True)["image"].count())

    for old in glob.glob(os.path.join(out_dir, "fig_*.png")):
        os.remove(old)
    for folder in ("validation", "density", "overlays"):
        shutil.rmtree(os.path.join(out_dir, folder), ignore_errors=True)
    val_df.to_csv(os.path.join(out_dir, "validation_per_image.csv"), index=False)
    val_summary.to_csv(os.path.join(out_dir, "validation_summary.csv"), index=False)
    density_df.to_csv(os.path.join(out_dir, "density_per_image.csv"), index=False)
    density_summary.to_csv(os.path.join(out_dir, "density_summary.csv"))
    comparison.to_csv(os.path.join(out_dir, "density_stats.csv"), index=False)

    for group in val_df["group"].unique():
        group_dir = os.path.join(out_dir, "validation", str(group))
        os.makedirs(group_dir, exist_ok=True)
        for metric in VALIDATION_METRICS:
            plot_validation(val_df, group, metric, class_order, os.path.join(group_dir, f"{metric}.png"))

    os.makedirs(os.path.join(out_dir, "density"), exist_ok=True)
    top = np.nanmax(density_df[["manual_density", "predicted_density"]].to_numpy())
    for source, file_name in (("manual", "manual.png"), ("predicted", "model.png")):
        plot_density(density_df, comparison, compare, source, top, os.path.join(out_dir, "density", file_name))

    counts = density_df.set_index("image")
    for name in test_names:
        if name not in gts or name not in counts.index:
            continue
        group_dir = os.path.join(out_dir, "overlays", str(counts.loc[name, "group"]))
        os.makedirs(group_dir, exist_ok=True)
        plot_overlay(name, settings["image_dir"], gts[name], preds[name], counts.loc[name], settings["colormap"],
                     os.path.join(group_dir, os.path.basename(name)))

    fmt = lambda v: f"{v:.3f}"
    print("\n=== Validation on held-out test images (mean per image) ===")
    print(val_summary[["group", "class", "n_images"] + [f"{m}_mean" for m in metrics]]
          .rename(columns=lambda c: c.replace("_mean", "")).to_string(index=False, float_format=fmt, na_rep="-"))
    print(f"\n=== Microvilli density per {DENSITY_UNIT:,} px of cell (median per group) ===")
    print(density_summary.to_string(float_format=fmt))
    print("\n=== Group comparison (Holm-adjusted) ===")
    print(comparison[["source", "comparison", "test", "p_value", "p_holm"]].to_string(index=False, float_format=lambda v: f"{v:.2g}"))
    print(f"\nAll tables and figures saved to {out_dir}")


if __name__ == "__main__":
    main()
