#!/usr/bin/env python

import argparse
import csv
import functools
import glob
import hashlib
import json
import os
from PIL import Image
from typing import Optional
import random

import numpy as np
import torch
import yaml
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import adjust_brightness
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI flags; each defaults to None so config.yaml values apply unless overridden."""
    parser = argparse.ArgumentParser(description="Train / fine-tune FAST UNet++ on TIRF images.")
    parser.add_argument("--config", default=None, help="Path to config.yaml. CLI flags below override its values.")
    parser.add_argument("--image-dir", default=None, help="Overrides config's image-dir (or converted-dir).")
    parser.add_argument("--mask-dir", default=None, help="Overrides config's mask-dir (or converted-dir).")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None, help="Number of classes incl. background.")
    parser.add_argument("--pretrained-weights", default=None, help="Checkpoint to fine-tune from (e.g. fast_model.pth).")
    parser.add_argument("--continue-from", default=None, help="A previous run's output dir (e.g. supervisely_run_3class) to "
                         "fine-tune from. Defaults --pretrained-weights to <continue-from>/model/best_model.pth if not set "
                         "explicitly. Does NOT affect which images are test -- that's always every dataset whose name ends "
                         "in _Test, independent of training history.")
    parser.add_argument("--k-folds", type=int, default=None, help="Number of cross-validation folds over the non-test images "
                         "(default 5), stratified by group (see config's `groups`) so every fold's train/val split has "
                         "roughly the same group mix. Each fold trains its own model (from the same starting checkpoint); "
                         "all k are kept and ensembled at inference.")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to 1e-4 with pretrained weights, else 1e-3.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only resolve the test set, assign every image its group "
                         "label and build the stratified folds: print the group x fold table, write group_labels.csv to "
                         "out-dir, and exit without training. Use it to check group assignment before a real run.")
    return parser


def resolve_continue_from_checkpoint(continue_from: str) -> Optional[str]:
    """The checkpoint to fine-tune from when continuing from a previous run's output dir."""
    model_dir = os.path.join(continue_from, "model")
    fold_paths = sorted(glob.glob(os.path.join(model_dir, "fold*_model.pth")))
    if not fold_paths:
        legacy = os.path.join(model_dir, "best_model.pth")
        return legacy if os.path.exists(legacy) else None

    state_dicts = [torch.load(p, map_location="cpu") for p in fold_paths]
    averaged = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        averaged[key] = torch.stack(tensors).mean(dim=0) if tensors[0].is_floating_point() else tensors[0]

    averaged_path = os.path.join(model_dir, "averaged_model.pth")
    torch.save(averaged, averaged_path)
    return averaged_path


def load_config(args: argparse.Namespace) -> dict:
    """Resolve settings with precedence CLI > config.yaml > defaults."""
    file_cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            file_cfg = yaml.safe_load(f) or {}

    def pick(cli_value, key, default=None):
        if cli_value is not None:
            return cli_value
        if file_cfg.get(key) is not None:
            return file_cfg[key]
        return default

    converted_dir = file_cfg.get("converted-dir")
    image_dir = pick(args.image_dir, "image-dir", converted_dir)
    mask_dir = pick(args.mask_dir, "mask-dir", converted_dir)
    if image_dir is None or mask_dir is None:
        raise ValueError("image/mask dirs not set: pass --image-dir/--mask-dir, or set image-dir/mask-dir or converted-dir in the config file")

    continue_from = pick(args.continue_from, "continue-from")
    pretrained_weights = args.pretrained_weights
    if pretrained_weights is None and continue_from:
        pretrained_weights = resolve_continue_from_checkpoint(continue_from)
    if pretrained_weights is None:
        pretrained_weights = file_cfg.get("pretrained-weights")
    lr = pick(args.lr, "lr", 1e-4 if pretrained_weights else 1e-3)

    classes = pick(args.classes, "classes")
    colormap_dict = file_cfg.get("colormap")
    if classes is None or not colormap_dict:
        raise ValueError("`classes` and `colormap` must be set in the config file")
    colormap = np.zeros((max(int(k) for k in colormap_dict) + 1, 3), dtype=np.uint8)
    for class_id, rgb in colormap_dict.items():
        colormap[int(class_id)] = rgb
    if len(colormap) != classes:
        raise ValueError(f"config has classes={classes} but colormap defines {len(colormap)} class ids (0..{len(colormap) - 1})")

    class_name_to_index = file_cfg.get("class-name-to-index") or {}
    cldice_class_names = file_cfg.get("cldice-classes") or []
    unknown = [n for n in cldice_class_names if n not in class_name_to_index]
    if unknown:
        raise ValueError(f"cldice-classes lists unknown class(es) {unknown}; class-name-to-index defines {list(class_name_to_index)}")

    return {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "out_dir": pick(args.out_dir, "out-dir", "supervisely_run"),
        "epochs": pick(args.epochs, "epochs", 100),
        "batch_size": pick(args.batch_size, "batch-size", 2),
        "k_folds": pick(args.k_folds, "k-folds", 5),
        "classes": classes,
        "pretrained_weights": pretrained_weights,
        "continue_from": continue_from,
        "lr": lr,
        "seed": pick(args.seed, "seed", 42),
        "colormap": colormap,
        "cldice_weight": file_cfg.get("cldice-weight", 0.0),
        "cldice_class_ids": [int(class_name_to_index[n]) for n in cldice_class_names],
        "cldice_skel_iters": file_cfg.get("cldice-skel-iters", 10),
        "groups": file_cfg.get("groups") or {},
    }


def _as_list(value) -> list:
    """None -> [], one path -> [path], a list -> itself."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


class AddGaussianNoise:
    """Adds N(mean, std) noise to an already-ToTensor'd image tensor, with probability p."""

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size(),device = tensor.device)*self.std + self.mean
            return tensor + noise
        return tensor


class RandomBrightness:
    """Randomly scales image brightness within brightness_factor_range, with probability p."""

    def __init__(self, brightness_factor_range: tuple[float, float] = (0.5, 1.5), p: float = 0.5):
        self.brightness_factor_range = brightness_factor_range
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            image = adjust_brightness(image, brightness_factor)
        return image


def list_dataset_images(root_dir) -> list[str]:
    """All '<dataset>/<file>.png' names under one or more converted roots."""
    names = []
    dataset_root = {}
    for root in _as_list(root_dir):
        if not os.path.isdir(root):
            continue
        for dataset in sorted(os.listdir(root)):
            images_subdir = os.path.join(root, dataset, "images")
            if not os.path.isdir(images_subdir):
                continue
            if dataset in dataset_root:
                raise ValueError(f"Dataset folder '{dataset}' exists in both {dataset_root[dataset]} and {root} -- "
                                 "rename one of the Supervisely datasets so every dataset name is unique across converted-dir.")
            dataset_root[dataset] = root
            for file in sorted(os.listdir(images_subdir)):
                if file.endswith(".png"):
                    names.append(f"{dataset}/{file}")
    return names


def _dataset_root(root_dir, dataset: str) -> str:
    """The converted root (of one or several) that holds `dataset`."""
    roots = _as_list(root_dir)
    for root in roots:
        if os.path.isdir(os.path.join(root, dataset)):
            return root
    return roots[0]


def resolve_image_path(image_dir, name: str) -> str:
    dataset, filename = name.split("/", 1)
    return os.path.join(_dataset_root(image_dir, dataset), dataset, "images", filename)


def resolve_mask_path(mask_dir, name: str) -> str:
    dataset, filename = name.split("/", 1)
    return os.path.join(_dataset_root(mask_dir, dataset), dataset, "masks", filename)


class TIRFDataset(Dataset):
    """Grayscale image / class-id mask pairs, addressed by '<dataset>/<file>.png' name."""

    def __init__(self, image_dir, mask_dir, image_transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.images = list_dataset_images(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, mask_tensor)."""
        img_name = self.images[idx]
        img_path = resolve_image_path(self.image_dir, img_name)
        mask_path = resolve_mask_path(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transforms(image)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


def file_content_hash(path: str) -> str:
    with Image.open(path) as im:
        return hashlib.md5(np.array(im.convert("L")).tobytes()).hexdigest()


def test_split(filenames: list[str], image_dir) -> list[str]:
    """Images in datasets named *_Test, after checking no test image is duplicated in a training dataset."""
    test_files = sorted(n for n in filenames if n.split("/", 1)[0].lower().endswith("_test"))
    if not test_files:
        print("WARNING: no dataset name ends in _Test -- nothing is held out as test.")

    test = set(test_files)
    by_hash = {}
    for n in filenames:
        by_hash.setdefault(file_content_hash(resolve_image_path(image_dir, n)), []).append(n)
    leaks = [names for names in by_hash.values() if any(n in test for n in names) and any(n not in test for n in names)]
    if leaks:
        raise SystemExit(f"{len(leaks)} test image(s) also appear in a training dataset (same pixels), e.g. {leaks[0]} -- "
                         "remove the duplicate from one of the datasets and re-run convert.sh.")
    return test_files


def assign_group(name: str, groups: dict) -> str:
    """Group of an image: the first config group whose pattern is in its filename."""
    lowered = os.path.basename(name).lower()
    for group, patterns in groups.items():
        if any(str(p).lower() in lowered for p in patterns):
            return group
    return "Unassigned"


def group_labels(names: list[str], groups: dict) -> list[str]:
    """Group label of every name, warning about unassigned or ambiguous filenames."""
    y = [assign_group(n, groups) for n in names]
    unassigned = [n for n, g in zip(names, y) if g == "Unassigned"]
    ambiguous = []
    for n in names:
        lowered = os.path.basename(n).lower()
        hits = [g for g, patterns in groups.items() if any(str(p).lower() in lowered for p in patterns)]
        if len(hits) > 1:
            ambiguous.append((n, hits))
    if groups and unassigned:
        print(f"WARNING: {len(unassigned)} image(s) match no group pattern (-> 'Unassigned'), e.g. {unassigned[:3]}")
    for n, hits in ambiguous:
        print(f"WARNING: {n} matches several groups {hits} -- assigned to '{hits[0]}' (first in config order)")
    return y


def make_folds(pool: list[str], k: int, seed: int, y: Optional[list[str]] = None) -> list[list[str]]:
    """Stratified k-fold split of pool by group label y."""
    if y is None:
        y = ["all"] * len(pool)
    label_of = dict(zip(pool, y))
    pool = sorted(pool)
    k = max(1, min(k, len(pool))) if pool else 1
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    position = 0
    for group in sorted(set(label_of.values())):
        members = [n for n in pool if label_of[n] == group]
        if len(members) < k:
            print(f"WARNING: group '{group}' has only {len(members)} non-test image(s) for {k} folds -- "
                  f"it is absent from {k - len(members)} fold(s)' val set")
        rng.shuffle(members)
        for name in members:
            folds[position % k].append(name)
            position += 1
    return [sorted(f) for f in folds]


def write_group_labels(path: str, names: list[str], y: list[str], test_files: list[str], folds: list[list[str]]) -> None:
    """Write image, group, split and val_fold for every image to a CSV."""
    val_fold_of = {n: i for i, fold in enumerate(folds) for n in fold}
    test = set(test_files)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "group", "split", "val_fold"])
        for n, g in zip(names, y):
            writer.writerow([n, g, "test" if n in test else "train+val", "" if n in test else val_fold_of[n]])


def print_group_table(y_of: dict, test_files: list[str], folds: list[list[str]]) -> None:
    """Print the number of images per group in the test set and each validation fold."""
    groups = sorted(set(y_of.values()))
    columns = ["test"] + [f"val{i}" for i in range(len(folds))] + ["total"]
    rows = []
    for g in groups:
        counts = [sum(y_of[n] == g for n in test_files)] + [sum(y_of[n] == g for n in fold) for fold in folds]
        rows.append([g] + counts + [sum(counts)])
    width = max([len("group")] + [len(g) for g in groups]) + 2
    print("\nImages per group (val<i> = fold i's validation set; its training set is the rest of the non-test pool):")
    print("group".ljust(width) + "".join(c.rjust(7) for c in columns))
    for row in rows:
        print(row[0].ljust(width) + "".join(str(v).rjust(7) for v in row[1:]))


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss, averaged over all classes (including background) and the batch."""
    pred = torch.softmax(pred, dim=1)
    nclass = pred.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=nclass).permute(0,3,1,2).float()
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2.0 * intersection + smooth)/(union + smooth)
    return 1 - dice.mean()


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Differentiable approximate erosion of a soft mask."""
    p1 = -nn.functional.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -nn.functional.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return nn.functional.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable soft skeleton of a (B, 1, H, W) mask (Shit et al., 2021)."""
    img1 = _soft_open(img)
    skeleton = nn.functional.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = nn.functional.relu(img - img1)
        skeleton = skeleton + nn.functional.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor, class_ids: list[int], iterations: int = 10, smooth: float = 1e-6) -> torch.Tensor:
    """Differentiable clDice loss over the given class ids."""
    pred_soft = torch.softmax(pred, dim=1)
    nclass = pred_soft.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=nclass).permute(0, 3, 1, 2).float()

    losses = []
    for class_id in class_ids:
        p = pred_soft[:, class_id : class_id + 1]
        t = target_one_hot[:, class_id : class_id + 1]
        skel_p = soft_skeletonize(p, iterations)
        skel_t = soft_skeletonize(t, iterations)
        t_prec = ((skel_p * t).sum(dim=(2, 3)) + smooth) / (skel_p.sum(dim=(2, 3)) + smooth)
        t_sens = ((skel_t * p).sum(dim=(2, 3)) + smooth) / (skel_t.sum(dim=(2, 3)) + smooth)
        cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + smooth)
        losses.append(1 - cl_dice.mean())
    return torch.stack(losses).mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, cldice_weight: float = 0.0,
                   cldice_class_ids: Optional[list[int]] = None, cldice_skel_iters: int = 10) -> torch.Tensor:
    """(1 - cldice_weight) * dice_loss + cldice_weight * soft_cldice_loss."""
    d = dice_loss(pred, target)
    if not cldice_weight or not cldice_class_ids:
        return d
    cl = soft_cldice_loss(pred, target, cldice_class_ids, cldice_skel_iters)
    return (1 - cldice_weight) * d + cldice_weight * cl


def build_model(classes: int, device: torch.device) -> nn.Module:
    """UNet++ (resnet34 encoder, 1 input channel) with the given number of classes."""
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights = "imagenet",
        in_channels=1,
        classes=classes,
    )
    return model.to(device)


class EnsembleModel(nn.Module):
    """Averages softmax probabilities across the k-fold member models."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.models], dim=0)
        return probs.mean(dim=0)


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load only the checkpoint tensors whose name and shape match the model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()

    compatible, skipped = {}, []
    for name, tensor in checkpoint.items():
        if name in model_state and model_state[name].shape == tensor.shape:
            compatible[name] = tensor
        else:
            skipped.append(name)

    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(checkpoint)} tensors from {checkpoint_path}")
    if skipped:
        print(f"skipped (missing or shape mismatch): {skipped}")
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, loss_fn) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> float:
    """Mean loss over loader without gradient updates."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            running_loss += loss_fn(outputs, masks).item()
    return running_loss / len(loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    loss_fn,
    writer=None,
) -> tuple[dict, float]:
    """Runs the full training loop for `epochs` epochs."""
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
        msg = f"Epoch {epoch + 1}/{epochs}  train loss {train_loss:.4f}"

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, loss_fn)
            if writer is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)
            msg += f"  val loss {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                msg += "  *"
        print(msg)

    if val_loader is None:
        best_state = copy.deepcopy(model.state_dict())
    return best_state, best_val_loss


def decode_segmentation_mask(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Map a class-id mask to an RGB image via colormap."""
    return colormap[mask]


def save_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: TIRFDataset,
    colormap: np.ndarray,
    out_dir: str,
    device: torch.device,
) -> None:
    """Save ground-truth and predicted RGB masks for every test image."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    filenames = test_dataset.images
    i = 0
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            masks = masks.numpy()
            for pred, mask in zip(preds, masks):
                name = filenames[i]
                i += 1
                safe_name = name.replace("/", "__")
                Image.fromarray(decode_segmentation_mask(mask, colormap)).save(os.path.join(out_dir, f"decoded_mask_{safe_name}"))
                Image.fromarray(decode_segmentation_mask(pred, colormap)).save(os.path.join(out_dir, f"decoded_pred_{safe_name}"))


def main() -> None:

    args = build_arg_parser().parse_args()
    config = load_config(args)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    out_dir = config["out_dir"]
    model_dir = os.path.join(out_dir, "model")
    predictions_dir = os.path.join(model_dir, "predictions")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
        RandomBrightness(brightness_factor_range=(0.8, 1.2), p=0.5)
    ])
    eval_transform = transforms.ToTensor()

    base_dataset = TIRFDataset(config['image_dir'], config["mask_dir"], eval_transform)
    filenames = sorted(base_dataset.images)

    test_files = test_split(filenames, config["image_dir"])
    pool = sorted(f for f in filenames if f not in set(test_files))

    y = group_labels(filenames, config["groups"])
    y_of = dict(zip(filenames, y))
    folds = make_folds(pool, config["k_folds"], config["seed"], [y_of[n] for n in pool])
    k = len(folds)
    print(f"Found {len(filenames)} images -> {k}-fold stratified CV over {len(pool)} (train+val), test={len(test_files)}")
    print_group_table(y_of, test_files, folds)

    os.makedirs(out_dir, exist_ok=True)
    write_group_labels(os.path.join(out_dir, "group_labels.csv"), filenames, y, test_files, folds)
    print(f"Group labels written to {os.path.join(out_dir, 'group_labels.csv')}")
    if args.dry_run:
        print("Dry run: stopping before training.")
        return
    os.makedirs(predictions_dir, exist_ok=True)

    def make_dataset(names, image_transforms):
        dataset = TIRFDataset(config["image_dir"], config["mask_dir"], image_transforms)
        dataset.images = list(names)
        return dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = config["batch_size"]

    loss_fn = functools.partial(combined_loss, cldice_weight=config["cldice_weight"],
                                 cldice_class_ids=config["cldice_class_ids"], cldice_skel_iters=config["cldice_skel_iters"])
    if config["cldice_weight"] and config["cldice_class_ids"]:
        print(f"clDice loss enabled: weight={config['cldice_weight']}, classes={config['cldice_class_ids']}")

    fold_results = []

    for fold_idx in range(k):
        val_files = folds[fold_idx]
        train_files = sorted(name for j, fold in enumerate(folds) if j != fold_idx for name in fold)
        print(f"\n--- Fold {fold_idx + 1}/{k}: train={len(train_files)}, val={len(val_files)} ---")

        train_dataset = make_dataset(train_files, train_transform)
        val_dataset = make_dataset(val_files, eval_transform) if val_files else None
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False) if val_dataset else None

        model = build_model(config["classes"], device)
        if config["pretrained_weights"]:
            load_pretrained_weights(model, config["pretrained_weights"], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        if fold_idx == 0:
            print(f"Learning rate: {config['lr']}")

        writer = SummaryWriter(os.path.join(out_dir, "logs", f"fold{fold_idx}"))
        fold_state, fold_val_loss = train(model, train_loader, val_loader, optimizer, device, config["epochs"], loss_fn, writer)
        writer.close()
        print(f"Fold {fold_idx + 1}/{k} best val loss: {fold_val_loss:.4f}")
        fold_results.append({"state": fold_state, "train": train_files, "val": val_files, "val_loss": fold_val_loss})

    val_losses = [r["val_loss"] for r in fold_results if r["val_loss"] != float("inf")]
    if val_losses:
        mean_loss = sum(val_losses) / len(val_losses)
        std_loss = (sum((v - mean_loss) ** 2 for v in val_losses) / len(val_losses)) ** 0.5
        print(f"\nCV val loss across {len(val_losses)} fold(s): {mean_loss:.4f} +/- {std_loss:.4f} "
              f"(mean +/- std -- a performance ESTIMATE only; every fold's model is kept and ensembled, none picked as \"best\")")

    split_hashes = {name: file_content_hash(resolve_image_path(config["image_dir"], name)) for name in filenames}
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({
            "folds": [{"train": r["train"], "val": r["val"], "val_loss": r["val_loss"]} for r in fold_results],
            "test": test_files,
            "groups": y_of,
            "hashes": split_hashes,
        }, f, indent=2)

    for fold_idx, r in enumerate(fold_results):
        torch.save(r["state"], os.path.join(model_dir, f"fold{fold_idx}_model.pth"))
    print(f"Saved {k} fold model(s) to {model_dir}/fold*_model.pth "
          f"(ensembled at inference -- see quantification.py/pseudo_label.py's load_model)")

    test_dataset = make_dataset(test_files, eval_transform) if test_files else None
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False) if test_dataset else None
    if test_loader is not None:
        members = [build_model(config["classes"], device) for _ in fold_results]
        for member, r in zip(members, fold_results):
            member.load_state_dict(r["state"])
        ensemble = EnsembleModel(members).to(device)
        ensemble.eval()
        save_test_predictions(ensemble, test_loader, test_dataset, config["colormap"], predictions_dir, device)
        print(f"Saved test predictions to {predictions_dir}")
    else:
        print("Not enough image held-out for testing - skipping test predictions.")

if __name__ == "__main__":
    main()
