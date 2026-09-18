#!/usr/bin/env python
# coding: utf-8
"""
Train / fine-tune the FAST UNet++ segmentation model on TIRF images.

This is a from-scratch rewrite (for learning purposes) that should still
follow the same overall shape as the original FAST train.py: augmentation ->
Dataset -> Dice loss -> UNet++ model -> train/val loop -> save best
checkpoint -> decode + save test predictions.

Fill in each function's body. Signatures + docstrings define the contract
each piece must satisfy so they compose correctly; the original train.py and
train_supervisely.py (a reference implementation of this same idea) are
there if you get stuck on a specific piece.

Suggested build + test order (don't write main() first):
  1. dice_loss              - pure function, easiest to unit-test standalone
  2. TIRFDataset (no aug)   - load one item, check image/mask shapes+dtypes
  3. split_dataset          - check it returns non-overlapping index lists
  4. augmentation classes   - wire into TIRFDataset, visualize a few outputs
  5. build_model / load_pretrained_weights
  6. train_one_epoch / evaluate / train
  7. decode_segmentation_mask / save_test_predictions
  8. load_config / build_arg_parser
  9. main() - wire it all together last
"""

import argparse
import os
from PIL import Image
from typing import Optional
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import adjust_brightness, crop, resize


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
    raise NotImplementedError


def load_config(args: argparse.Namespace) -> dict:
    """Resolve final settings from (in precedence order) CLI args > a
    config.yaml file (path given by args.config) > hardcoded defaults.

    Must resolve image_dir/mask_dir either from explicit config keys or by
    joining a `converted-dir` config key with "images"/"masks". Must also
    resolve `lr`: if unset anywhere, use 1e-4 when pretrained_weights is
    set, else 1e-3 (fine-tuning needs a gentler learning rate than training
    from scratch, or you overwrite the pretrained weights' useful features
    in the first few steps).

    Returns a dict with (at least) keys: image_dir, mask_dir, out_dir,
    epochs, batch_size, classes, pretrained_weights, lr, seed, colormap
    (an (n_classes, 3) uint8 np.ndarray).
    """
    raise NotImplementedError


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
            mask = resize(mask, self.resize_to)
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
        

class RandomBrightness:
    """Randomly scales image brightness within brightness_factor_range, with
    probability p. Image-only, same reasoning as AddGaussianNoise.
    """

    def __init__(self, brightness_factor_range: tuple[float, float] = (0.5, 1.5), p: float = 0.5):
        self.brightness_factor_range = brightness_factor_range
        self.p = p

    def __call__(self, image):
        if random.random < self.p:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            image = adjust_brightness(image, brightness_factor)
        return image


# --------------------------------------------------------------------------
# 3. Dataset
# --------------------------------------------------------------------------

class TIRFDataset(Dataset):
    """Loads (image, mask) PNG pairs from image_dir/mask_dir, matched by
    identical filename. Images are opened as single-channel ("L" mode)
    grayscale. Masks are ALSO single-channel PNGs, but their pixel values
    ARE the class index directly (0=background, 1..N-1=foreground classes)
    -- not a picture to look at, a label map. Watch the dtype: after
    ToTensor() a mask gets scaled into [0,1] float, so you must multiply
    back by 255 and cast to long to recover the original integer class ids
    before computing any loss.
    """

    def __init__(self, image_dir: str, mask_dir: str, image_transforms=None, random_crop: Optional[RandomCropPair] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.random_crop = random_crop
        self.images = []
        for file in os.listdir(image_dir):
            if file.endswith(".png"):
                self.images.append(file)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image_tensor, mask_tensor). mask_tensor must be dtype
        long with raw class-index values (see class docstring above)."""
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # load image and convert to greyscale
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.random_crop:
            image, mask = self.random_crop(image, mask)


        image = self.image_transforms(image)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask




def split_dataset(n_images: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Deterministically (via seed) shuffle range(n_images) and split into
    train/val/test index lists.

    With very few labeled images, a naive round(n * frac) split can leave
    val or test empty, which breaks downstream code that assumes a val/test
    loader exists. Guarantee at least 1 image in val and test once
    n_images >= 3; below that, val/test can be empty and callers must
    handle that (see train()/evaluate()'s Optional loaders).
    """
    raise NotImplementedError


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
    you actually care about here.

    Returns a scalar tensor (differentiable, so this can be .backward()'d).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# 5. Model
# --------------------------------------------------------------------------

def build_model(classes: int, device: torch.device) -> nn.Module:
    """Construct the segmentation model: smp.UnetPlusPlus(
    encoder_name='resnet34', encoder_weights='imagenet', in_channels=1,
    classes=classes), moved to device. in_channels=1 because these are
    grayscale TIRF images, not RGB.
    """
    raise NotImplementedError


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load checkpoint_path's state_dict into model IN PLACE, copying over
    only tensors whose name AND shape both match model's own state_dict.

    This matters because fast_model.pth was trained with classes=5
    (includes Filopodia); if your model here uses classes=4, the final
    segmentation_head layer's shape won't match and must be skipped (left
    at its random init) rather than crashing the whole load. Print a
    summary of how many tensors loaded vs. were skipped, and which ones
    were skipped, so a silent shape mismatch doesn't go unnoticed.

    Returns model (mutated in place, but returned for convenience).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# 6. Training loop
# --------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Runs one training epoch over loader: model.train(), then for each
    batch, zero_grad -> forward -> dice_loss -> backward -> step. Returns
    the mean loss across all batches.
    """
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Runs one pass over loader with model.eval() and torch.no_grad(), and
    returns the mean dice_loss across all batches. Used for both val-loss
    tracking during training and a final test-loss number.
    """
    raise NotImplementedError


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    writer=None,
) -> dict:
    """Runs the full training loop for `epochs` epochs.

    Each epoch: train_one_epoch(...), then if val_loader is not None,
    evaluate(...) and -- only when val loss improves on the best-so-far --
    keep a deep copy of model.state_dict() as the running "best" checkpoint.
    If val_loader is None (too few images to hold any out), there's nothing
    to compare against, so just keep the latest epoch's weights every time.

    If writer is given (a torch.utils.tensorboard.SummaryWriter), log
    train/val loss per epoch via writer.add_scalar(...).

    Returns the best (or final, if no val_loader) state_dict -- this is
    what main() should torch.save().
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# 7. Visualization / test predictions
# --------------------------------------------------------------------------

def decode_segmentation_mask(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Maps a (H, W) array of integer class indices to an (H, W, 3) uint8
    RGB image via colormap (shape (n_classes, 3)). Used to actually SEE a
    mask -- raw class-index pixel values (0, 1, 2, 3) look solid black in
    any normal image viewer.
    """
    raise NotImplementedError


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
    raise NotImplementedError


# --------------------------------------------------------------------------
# 8. Entrypoint
# --------------------------------------------------------------------------

def main() -> None:
    """Wire everything together:
      1. build_arg_parser().parse_args(), load_config(args)
      2. seed random/torch with config['seed'] (reproducible split+init)
      3. build one base TIRFDataset (no transforms) just to count images
         and get filenames; split_dataset(...) on that count
      4. build train/val/test TIRFDataset instances (train gets
         augmentation + random_crop, val/test don't), sliced to their
         split's filenames, wrapped in DataLoaders
      5. build_model(...); if config['pretrained_weights'],
         load_pretrained_weights(...)
      6. optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
      7. best_state = train(...)
      8. torch.save(best_state, .../best_model.pth)
      9. model.load_state_dict(best_state); save_test_predictions(...)
         (skip if the test split ended up empty)
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
