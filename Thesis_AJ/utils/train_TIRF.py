#!/usr/bin/env python
# coding: utf-8

import argparse
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
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import adjust_brightness, crop, resize
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------
# 1. Config resolution
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI flags: --config, --image-dir, --mask-dir, --out-dir,
    --epochs, --batch-size, --classes, --pretrained-weights, --lr, --seed.

    Every flag except --config should default to None (not a "real"
    default like 100 or 4) so load_config() can distinguish "user didn't
    pass this" from "user explicitly passed this value" and fall through
    to config.yaml, then to a hardcoded default, in that order.
    """
    parser = argparse.ArgumentParser(description="Train / fine-tune FAST UNet++ on TIRF images.")
    parser.add_argument("--config", default=None, help="Path to config.yaml. CLI flags below override its values.")
    parser.add_argument("--image-dir", default=None, help="Overrides config's image-dir (or converted-dir).")
    parser.add_argument("--mask-dir", default=None, help="Overrides config's mask-dir (or converted-dir).")
    parser.add_argument("--test-source-dir", default=None, help="Folder defining the pinned test set (e.g. Images/CD8AA_Cleaned/Test) "
                         "-- every image there (once labeled and converted) is permanently held out of training/validation.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None, help="Number of classes incl. background.")
    parser.add_argument("--pretrained-weights", default=None, help="Checkpoint to fine-tune from (e.g. fast_model.pth).")
    parser.add_argument("--continue-from", default=None, help="A previous run's output dir (e.g. supervisely_run_3class) to "
                         "fine-tune from. Defaults --pretrained-weights to <continue-from>/model/best_model.pth if not set "
                         "explicitly. Does NOT affect which images are test -- that's always test-source-dir (see below), "
                         "independent of training history.")
    parser.add_argument("--k-folds", type=int, default=None, help="Number of cross-validation folds over the non-test images "
                         "(default 5). Each fold trains its own model (from the same starting checkpoint); the fold with the "
                         "lowest val loss is kept as best_model.pth. This is what lets every non-test image contribute to "
                         "training eventually, instead of a fixed val subset never being trained on.")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to 1e-4 with pretrained weights, else 1e-3.")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def resolve_continue_from_checkpoint(continue_from: str) -> Optional[str]:
    """The checkpoint to fine-tune from when continuing from a previous
    run's output dir.

    A run now saves one model/fold{i}_model.pth per k-fold CV fold (see
    EnsembleModel) instead of a single best_model.pth -- averages their
    weights into one starting checkpoint rather than privileging one fold as
    "the" previous model. Safe here because every fold was fine-tuned from
    the SAME common ancestor checkpoint for a modest number of epochs, so
    they haven't diverged into different loss-landscape basins -- this is
    the "model soup" approach (Wortsman et al., 2022). The averaged
    checkpoint is cached to model/averaged_model.pth so it's visible on disk
    and not recomputed every run.

    Falls back to a plain model/best_model.pth for an older, pre-ensembling
    run. Returns None if continue_from has neither.
    """
    model_dir = os.path.join(continue_from, "model")
    fold_paths = sorted(glob.glob(os.path.join(model_dir, "fold*_model.pth")))
    if not fold_paths:
        legacy = os.path.join(model_dir, "best_model.pth")
        return legacy if os.path.exists(legacy) else None

    state_dicts = [torch.load(p, map_location="cpu") for p in fold_paths]
    averaged = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        # Only average floating-point weights; integer buffers (e.g.
        # BatchNorm's num_batches_tracked) aren't meaningful to average, so
        # just keep the first fold's value.
        averaged[key] = torch.stack(tensors).mean(dim=0) if tensors[0].is_floating_point() else tensors[0]

    averaged_path = os.path.join(model_dir, "averaged_model.pth")
    torch.save(averaged, averaged_path)
    return averaged_path


def load_config(args: argparse.Namespace) -> dict:
    """Resolve final settings from (in precedence order) CLI args > a
    config.yaml file (path given by args.config) > hardcoded defaults.

    Must resolve image_dir/mask_dir either from explicit config keys or by
    joining a `converted-dir` config key with "images"/"masks". Must also
    resolve `lr`: if unset anywhere, use 1e-4 when pretrained_weights is
    set, else 1e-3 (fine-tuning needs a gentler learning rate than training
    from scratch, or you overwrite the pretrained weights' useful features
    in the first few steps).

    Returns a dict with (at least) keys: image_dir, mask_dir, test_source_dir,
    out_dir, epochs, batch_size, k_folds, classes, pretrained_weights,
    continue_from, lr, seed, colormap (an (n_classes, 3) uint8 np.ndarray).
    """
    file_cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            file_cfg = yaml.safe_load(f) or {}

    def pick(cli_value, key, default=None):
        # `is not None` (not truthiness) so legitimate values like seed=0 aren't skipped.
        if cli_value is not None:
            return cli_value
        if file_cfg.get(key) is not None:
            return file_cfg[key]
        return default

    # image_dir/mask_dir both point at converted-dir itself (NOT
    # converted-dir/images or /masks): convert_supervisely_annotations.py
    # writes one <converted-dir>/<dataset>/{images,masks,preview} subfolder
    # per Supervisely dataset, mirroring project-dir's own structure, so
    # there's no single flat images/masks folder to point at. See
    # list_dataset_images/resolve_image_path/resolve_mask_path below for how
    # a "name" (always "<dataset>/<filename>.png") maps to an actual path.
    converted_dir = file_cfg.get("converted-dir")
    image_dir = pick(args.image_dir, "image-dir", converted_dir)
    mask_dir = pick(args.mask_dir, "mask-dir", converted_dir)
    if image_dir is None or mask_dir is None:
        raise ValueError("image/mask dirs not set: pass --image-dir/--mask-dir, or set image-dir/mask-dir or converted-dir in the config file")

    test_source_dir = pick(args.test_source_dir, "test-source-dir")
    if test_source_dir is None:
        raise ValueError("test-source-dir not set: pass --test-source-dir, or set test-source-dir in the config file "
                          "(the folder defining the pinned test set, e.g. Images/CD8AA_Cleaned/Test)")

    continue_from = pick(args.continue_from, "continue-from")
    # Precedence here is CLI flag > --continue-from's own checkpoint > config's
    # static pretrained-weights default -- deliberately NOT the usual CLI >
    # config precedence, since config.yaml's pretrained-weights is normally
    # fast_model.pth (for a fresh run) and --continue-from should override
    # that stale default, not lose to it. An explicit --pretrained-weights
    # still wins over --continue-from, for the rare case you want to pin a
    # previous run's test split but deliberately start from a different
    # checkpoint.
    pretrained_weights = args.pretrained_weights
    if pretrained_weights is None and continue_from:
        pretrained_weights = resolve_continue_from_checkpoint(continue_from)
    if pretrained_weights is None:
        pretrained_weights = file_cfg.get("pretrained-weights")
    lr = pick(args.lr, "lr", 1e-4 if pretrained_weights else 1e-3)

    # The class setup is defined only in the config (nothing class-specific is
    # hardcoded here), so changing the classes later is a config-only edit.
    classes = pick(args.classes, "classes")
    colormap_dict = file_cfg.get("colormap")
    if classes is None or not colormap_dict:
        raise ValueError("`classes` and `colormap` must be set in the config file")
    colormap = np.zeros((max(int(k) for k in colormap_dict) + 1, 3), dtype=np.uint8)
    for class_id, rgb in colormap_dict.items():
        colormap[int(class_id)] = rgb
    if len(colormap) != classes:
        raise ValueError(f"config has classes={classes} but colormap defines {len(colormap)} class ids (0..{len(colormap) - 1})")

    # clDice loss (config-only, no CLI flags -- see combined_loss): which
    # classes it applies to is resolved from class-name-to-index, same
    # mapping convert_supervisely_annotations.py/quantification.py use.
    class_name_to_index = file_cfg.get("class-name-to-index") or {}
    cldice_class_names = file_cfg.get("cldice-classes") or []
    unknown = [n for n in cldice_class_names if n not in class_name_to_index]
    if unknown:
        raise ValueError(f"cldice-classes lists unknown class(es) {unknown}; class-name-to-index defines {list(class_name_to_index)}")

    return {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "test_source_dir": test_source_dir,
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
    }


# --------------------------------------------------------------------------
# 2. Data augmentation
# --------------------------------------------------------------------------

class RandomCropPair:
    """
    Crops image randomly to prevent overfit. 
    """

    def __init__(self, crop_size: tuple[int, int], resize_to: Optional[tuple[int, int]] = None, p: float = 0.5):
        self.crop_size = crop_size
        self.resize_to = resize_to
        self.p = p

    def __call__(self, image, mask):
        """Returns (image, mask), both PIL Images, possibly cropped+resized."""
        if random.random() < self.p:
            i,j,h,w = self.get_params(image, self.crop_size)
            image = crop(image,i,j,h,w)
            mask = crop(mask, i,j,h,w)
        if self.resize_to:
            image = resize(image, self.resize_to)
            # NEAREST for the mask: bilinear would blend class ids (e.g. 1 and 2 -> 1.5) at boundaries.
            mask = resize(mask, self.resize_to, interpolation=InterpolationMode.NEAREST)
        return image, mask

    @staticmethod
    def get_params(image, crop_size):
        width, height = image.size 
        crop_height, crop_width = crop_size
        if width < crop_width or height < crop_height:
            raise ValueError(f"crop size ({crop_width,crop_height}) must be smaller than image size ({width, height})")
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return top, left, crop_height, crop_width


class AddGaussianNoise:
    """Adds N(mean, std) noise to an already-ToTensor'd image tensor, with
    probability p. Image-only: never apply to the mask -- its pixel values
    are class indices (0, 1, 2, 3...), not intensities, and noise would
    corrupt the labels.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            # produce a tensor with the shape of image.
            # Every individual pixel gets its own independently-drawn random value (noise)
            noise = torch.randn(tensor.size(),device = tensor.device)*self.std + self.mean
            return tensor + noise
        return tensor


class RandomBrightness:
    """Randomly scales image brightness within brightness_factor_range, with
    probability p. Image-only, same reasoning as AddGaussianNoise.
    """

    def __init__(self, brightness_factor_range: tuple[float, float] = (0.5, 1.5), p: float = 0.5):
        self.brightness_factor_range = brightness_factor_range
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            image = adjust_brightness(image, brightness_factor)
        return image


# --------------------------------------------------------------------------
# 3. Dataset
# --------------------------------------------------------------------------

def list_dataset_images(root_dir: str) -> list[str]:
    """Every image "name" under root_dir, across every per-Supervisely-dataset
    subfolder convert_supervisely_annotations.py writes (root_dir/<dataset>/
    images/<filename>.png). A name is always "<dataset>/<filename>.png" --
    unique by construction, since each dataset is its own namespace -- and is
    what resolve_image_path/resolve_mask_path expect."""
    names = []
    if not os.path.isdir(root_dir):
        return names
    for dataset in sorted(os.listdir(root_dir)):
        images_subdir = os.path.join(root_dir, dataset, "images")
        if not os.path.isdir(images_subdir):
            continue
        for file in sorted(os.listdir(images_subdir)):
            if file.endswith(".png"):
                names.append(f"{dataset}/{file}")
    return names


def resolve_image_path(image_dir: str, name: str) -> str:
    dataset, filename = name.split("/", 1)
    return os.path.join(image_dir, dataset, "images", filename)


def resolve_mask_path(mask_dir: str, name: str) -> str:
    dataset, filename = name.split("/", 1)
    return os.path.join(mask_dir, dataset, "masks", filename)


class TIRFDataset(Dataset):
    """Loads (image, mask) PNG pairs from image_dir/mask_dir, matched by
    identical "<dataset>/<filename>.png" name (see list_dataset_images).
    Images are opened as single-channel ("L" mode) grayscale. Masks are ALSO
    single-channel PNGs, but their pixel values ARE the class index directly
    (0=background, 1..N-1=foreground classes)
    """

    def __init__(self, image_dir: str, mask_dir: str, image_transforms, random_crop: Optional[RandomCropPair] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.random_crop = random_crop
        self.images = list_dataset_images(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image_tensor, mask_tensor). mask_tensor must be dtype
        long with raw class-index values (see class docstring above)."""
        img_name = self.images[idx]
        img_path = resolve_image_path(self.image_dir, img_name)
        mask_path = resolve_mask_path(self.mask_dir, img_name)

        # load image and convert to greyscale
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.random_crop:
            image, mask = self.random_crop(image, mask)

        image = self.image_transforms(image)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask



def file_content_hash(path: str) -> str:
    with Image.open(path) as im:
        return hashlib.md5(np.array(im.convert("L")).tobytes()).hexdigest()


def resolve_pinned_test(filenames: list[str], image_dir: str, test_source_dir: str) -> list[str]:
    source_names = [n for n in os.listdir(test_source_dir) if n.endswith(".png")]
    source_hashes = {file_content_hash(os.path.join(test_source_dir, n)) for n in source_names}

    current_hash_to_name = {file_content_hash(resolve_image_path(image_dir, n)): n for n in filenames}

    test_files = sorted({current_hash_to_name[h] for h in source_hashes if h in current_hash_to_name})
    not_yet = len(source_hashes) - len(test_files)
    if not_yet:
        print(f"Note: {not_yet} image(s) in {test_source_dir} not yet labeled/converted -- not held out as test yet.")

    return test_files


def make_folds(pool: list[str], k: int, seed: int) -> list[list[str]]:
    """Shuffles pool and splits it into k roughly-equal folds (filename
    lists, each differing in size by at most 1). k is clamped to
    [1, len(pool)] -- each fold needs at least 1 image to be a usable val
    set, so k shrinks automatically if the pool is smaller than requested.
    """
    pool = sorted(pool)
    k = max(1, min(k, len(pool))) if pool else 1
    indices = list(range(len(pool)))
    random.Random(seed).shuffle(indices)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(pool[idx])
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------
# 4. Loss
# --------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss, averaged over all classes (including background) and
    the batch.

    Args:
        pred: raw model output (logits, pre-softmax), shape (B, C, H, W).
        target: class-index mask, shape (B, H, W), dtype long.
        smooth: added to numerator/denominator to avoid divide-by-zero when
            a class is absent from both pred and target in a sample.

    Why Dice instead of plain cross-entropy: these images are mostly
    background pixels, so pixel-wise CE/accuracy is dominated by trivially
    getting background right and barely penalizes missing a thin filopodia
    strand. Dice directly measures mask overlap per class, which is what
    we actually care about here.

    Returns a scalar tensor (differentiable, so this can be .backward()'d).
    """
    pred = torch.softmax(pred, dim=1)
    nclass = pred.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=nclass).permute(0,3,1,2).float()
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2.0 * intersection + smooth)/(union + smooth)
    return 1 - dice.mean()


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Approximate binary erosion of a soft (0..1) mask: a 3x1 then 1x3
    min-pool (there's no torch min-pool, so -maxpool(-img) instead)."""
    p1 = -nn.functional.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -nn.functional.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return nn.functional.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable approximation of morphological skeletonization (the
    "soft-skeleton" from Shit et al., 2021's clDice paper), on a (B, 1, H, W)
    soft (0..1) mask. Repeatedly erodes the mask, keeping whatever each
    erosion step is "about to delete" (img - open(img)) as part of the
    skeleton -- the same idea as classic erode-until-gone thinning, just with
    torch ops so it's differentiable end to end (unlike quantification.py's
    skimage.skeletonize, which can't be backpropped through)."""
    img1 = _soft_open(img)
    skeleton = nn.functional.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = nn.functional.relu(img - img1)
        skeleton = skeleton + nn.functional.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor, class_ids: list[int], iterations: int = 10, smooth: float = 1e-6) -> torch.Tensor:
    """Soft clDice loss (Shit et al., 2021) -- the differentiable,
    training-time counterpart to quantification.py's cl_dice eval metric.
    Instead of comparing raw pixel overlap, compares each mask's soft
    skeleton against the OTHER mask's full region, so the loss rewards
    getting a thin structure's connectivity/path right, independent of exact
    boundary placement. Only computed for `class_ids` (config.yaml's
    cldice-classes) -- forcing a skeleton-based loss onto a blobby class
    (e.g. Lamellipodia) doesn't mean much and would just add noise.

    Args: pred (B, C, H, W) logits, target (B, H, W) long class indices.
    Returns a scalar tensor (1 - clDice, so 0 is perfect, same convention as
    dice_loss), averaged over class_ids and the batch.
    """
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
    """(1 - cldice_weight) * dice_loss + cldice_weight * soft_cldice_loss.
    cldice_weight=0 (or no cldice_class_ids, i.e. config.yaml's
    cldice-classes is empty/unset) reduces exactly to plain dice_loss -- a
    drop-in replacement with zero effect on a config that doesn't set
    cldice-weight."""
    d = dice_loss(pred, target)
    if not cldice_weight or not cldice_class_ids:
        return d
    cl = soft_cldice_loss(pred, target, cldice_class_ids, cldice_skel_iters)
    return (1 - cldice_weight) * d + cldice_weight * cl


# --------------------------------------------------------------------------
# 5. Model
# --------------------------------------------------------------------------

def build_model(classes: int, device: torch.device) -> nn.Module:
    """Construct the segmentation model: smp.UnetPlusPlus(
    encoder_name='resnet34', encoder_weights='imagenet', in_channels=1,
    classes=classes), moved to device. in_channels=1 because these are
    grayscale TIRF images, not RGB.
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights = "imagenet",
        in_channels=1,
        classes=classes,
    )
    return model.to(device)


class EnsembleModel(nn.Module):
    """Averages predicted class probabilities across k-fold CV's member
    models, instead of deploying a single "best fold" model.

    Why: with few images per fold, a fold's val loss is mostly sampling
    noise (which specific images it happened to get), so picking the fold
    with the lowest val loss is largely picking the luckiest split, not the
    best model -- a form of selection bias from using CV to choose among k
    models rather than to estimate generalization error. Ensembling uses
    every fold's information instead of gambling on one.

    forward(x) returns averaged softmax probabilities (not logits) -- fine
    for inference (argmax is unaffected, see quantification.predict_mask),
    but never pass this through dice_loss/combined_loss, which expect raw
    logits and apply their own softmax.
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.models], dim=0)
        return probs.mean(dim=0)


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load checkpoint_path's state_dict into model IN PLACE, copying over
    only tensors whose name AND shape both match model's own state_dict.

    This matters because fast_model.pth was trained with classes=5
    (includes Filopodia); our model here uses classes=4, the final
    segmentation_head layer's shape won't match and must be skipped (left
    at its random init) rather than crashing the whole load. Print a
    summary of how many tensors loaded vs. were skipped, and which ones
    were skipped, so a silent shape mismatch doesn't go unnoticed.

    Returns model (mutated in place, but returned for convenience).
    """
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


# --------------------------------------------------------------------------
# 6. Training loop
# --------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, loss_fn) -> float:
    """Runs one training epoch over loader: model.train(), then for each
    batch, zero_grad -> forward -> loss_fn -> backward -> step. Returns
    the mean loss across all batches.
    """
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
    """Runs one pass over loader with model.eval() and torch.no_grad(), and
    returns the mean loss_fn across all batches. Used for both val-loss
    tracking during training and a final test-loss number.
    """
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
    """Runs the full training loop for `epochs` epochs.

    Each epoch: train_one_epoch(...), then if val_loader is not None,
    evaluate(...) and -- only when val loss improves on the best-so-far --
    keep a deep copy of model.state_dict() as the running "best" checkpoint.
    If val_loader is None (too few images to hold any out), there's nothing
    to compare against, so just keep the latest epoch's weights every time.

    If writer is given (a torch.utils.tensorboard.SummaryWriter), log
    train/val loss per epoch via writer.add_scalar(...).

    Returns (best_state, best_val_loss) -- the best (or final, if no
    val_loader) state_dict, and the val loss it achieved (float('inf') if
    there was no val_loader to score against). main() uses best_val_loss to
    compare folds under k-fold cross-validation and pick the best one.
    """
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

# --------------------------------------------------------------------------
# 7. Visualization / test predictions
# --------------------------------------------------------------------------

def decode_segmentation_mask(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Maps a (H, W) array of integer class indices to an (H, W, 3) uint8
    RGB image via colormap (shape (n_classes, 3)). Used to actually SEE a
    mask -- raw class-index pixel values (0, 1, 2, 3) look solid black in
    any normal image viewer.
    """
    return colormap[mask]


def save_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: TIRFDataset,
    colormap: np.ndarray,
    out_dir: str,
    device: torch.device,
) -> None:
    """For every image in test_loader (images the model never saw during
    training OR validation), run inference, decode both the ground-truth
    mask and the predicted mask to RGB via decode_segmentation_mask, and
    save both PNGs into out_dir -- so results can be checked visually, not
    just trusted from a loss number.
    """
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
                safe_name = name.replace("/", "__")  # name is "<dataset>/<filename>.png"; flatten for a single output folder
                Image.fromarray(decode_segmentation_mask(mask, colormap)).save(os.path.join(out_dir, f"decoded_mask_{safe_name}"))
                Image.fromarray(decode_segmentation_mask(pred, colormap)).save(os.path.join(out_dir, f"decoded_pred_{safe_name}"))



# --------------------------------------------------------------------------
# 8. Entrypoint
# --------------------------------------------------------------------------

def main() -> None:

    args = build_arg_parser().parse_args()
    config = load_config(args)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    out_dir = config["out_dir"]
    model_dir = os.path.join(out_dir, "model")
    predictions_dir = os.path.join(model_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # --- data ---
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
        RandomBrightness(brightness_factor_range=(0.8, 1.2), p=0.5)
    ])
    eval_transform = transforms.ToTensor()

    base_dataset = TIRFDataset(config['image_dir'], config["mask_dir"], eval_transform)
    filenames = sorted(base_dataset.images)

    # Test is defined entirely by test_source_dir (e.g. Images/CD8AA_Cleaned/Test)
    test_files = resolve_pinned_test(filenames, config["image_dir"], config["test_source_dir"])
    pool = sorted(f for f in filenames if f not in set(test_files))
    folds = make_folds(pool, config["k_folds"], config["seed"])
    k = len(folds)
    print(f"Found {len(filenames)} images -> {k}-fold CV over {len(pool)} (train+val), test={len(test_files)}")

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

    # --- k-fold cross-validation: train k models (same starting checkpoint,
    # different train/val split each). ALL k models are kept and ENSEMBLED
    # at inference (see EnsembleModel) rather than picking the single best
    # fold's val loss -- with this few images per fold, "best" fold is
    # mostly luck (which images that fold's small val split happened to
    # get), so picking one and discarding the rest would just be gambling on
    # that noise. This also still gets every non-test image trained on (each
    # is train in k-1 folds, val in the remaining 1).
    fold_results = []  # [{"state", "train", "val", "val_loss"}, ...]

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

    # Record every fold's own split (not a flat "train"/"val" -- every pool
    # image is val in exactly one fold and train in the rest, so there's no
    # single held-out val set anymore now that all folds are ensembled),
    # plus a content hash per image so a future --continue-from run can
    # still find a pinned test image even if it's since been renamed (see
    # resolve_pinned_test). quantification.py reads this file to know
    # exactly which images were held out for THIS model.
    split_hashes = {name: file_content_hash(resolve_image_path(config["image_dir"], name)) for name in filenames}
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({
            "folds": [{"train": r["train"], "val": r["val"], "val_loss": r["val_loss"]} for r in fold_results],
            "test": test_files,
            "hashes": split_hashes,
        }, f, indent=2)

    for fold_idx, r in enumerate(fold_results):
        torch.save(r["state"], os.path.join(model_dir, f"fold{fold_idx}_model.pth"))
    print(f"Saved {k} fold model(s) to {model_dir}/fold*_model.pth "
          f"(ensembled at inference -- see quantification.py/pseudo_label.py's load_model)")

    # --- test prediction using the full ensemble ---
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
