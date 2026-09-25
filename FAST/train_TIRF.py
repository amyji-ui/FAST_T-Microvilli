#!/usr/bin/env python
# coding: utf-8

import argparse
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
    parser.add_argument("--image-dir", default=None, help="Overrides config's image-dir (or converted-dir/images).")
    parser.add_argument("--mask-dir", default=None, help="Overrides config's mask-dir (or converted-dir/masks).")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None, help="Number of classes incl. background.")
    parser.add_argument("--pretrained-weights", default=None, help="Checkpoint to fine-tune from (e.g. fast_model.pth).")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to 1e-4 with pretrained weights, else 1e-3.")
    parser.add_argument("--seed", type=int, default=None)
    return parser


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

    converted_dir = file_cfg.get("converted-dir")
    image_dir = pick(args.image_dir, "image-dir", os.path.join(converted_dir, "images") if converted_dir else None)
    mask_dir = pick(args.mask_dir, "mask-dir", os.path.join(converted_dir, "masks") if converted_dir else None)
    if image_dir is None or mask_dir is None:
        raise ValueError("image/mask dirs not set: pass --image-dir/--mask-dir, or set image-dir/mask-dir or converted-dir in the config file")

    pretrained_weights = pick(args.pretrained_weights, "pretrained-weights")
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

    return {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "out_dir": pick(args.out_dir, "out-dir", "supervisely_run"),
        "epochs": pick(args.epochs, "epochs", 100),
        "batch_size": pick(args.batch_size, "batch-size", 2),
        "classes": classes,
        "pretrained_weights": pretrained_weights,
        "lr": lr,
        "seed": pick(args.seed, "seed", 42),
        "colormap": colormap,
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

class TIRFDataset(Dataset):
    """Loads (image, mask) PNG pairs from image_dir/mask_dir, matched by
    identical filename. Images are opened as single-channel ("L" mode)
    grayscale. Masks are ALSO single-channel PNGs, but their pixel values
    ARE the class index directly (0=background, 1..N-1=foreground classes)
    """

    def __init__(self, image_dir: str, mask_dir: str, image_transforms, random_crop: Optional[RandomCropPair] = None):
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
    indices = list(range(n_images))
    rng = random.Random(seed); rng.shuffle(indices)

    train_n = max(1,round(n_images*train_frac))
    val_n = max(1,round(n_images*val_frac))
    test_n = max(0,n_images - train_n - val_n)
    if test_n == 0 and n_images >= 3:
        train_n -= 1
        test_n = 1

    train_list, val_list, test_list = indices[:train_n],indices[train_n:train_n+val_n], indices[train_n+val_n:train_n+val_n+test_n]
    return train_list, val_list, test_list


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

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Runs one training epoch over loader: model.train(), then for each
    batch, zero_grad -> forward -> dice_loss -> backward -> step. Returns
    the mean loss across all batches.
    """
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Runs one pass over loader with model.eval() and torch.no_grad(), and
    returns the mean dice_loss across all batches. Used for both val-loss
    tracking during training and a final test-loss number.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            running_loss += dice_loss(outputs, masks).item()
    return running_loss / len(loader)


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
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
        msg = f"Epoch {epoch + 1}/{epochs}  train loss {train_loss:.4f}"

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
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
    return best_state

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
                Image.fromarray(decode_segmentation_mask(mask, colormap)).save(os.path.join(out_dir, f"decoded_mask_{name}"))
                Image.fromarray(decode_segmentation_mask(pred, colormap)).save(os.path.join(out_dir, f"decoded_pred_{name}"))



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
    train_idx, val_idx, test_idx = split_dataset(len(filenames), seed=config["seed"])
    print(f"Found {len(filenames)} images -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Record which images went where. The split depends on how many images
    # exist, so it changes when the dataset grows; quantification.py reads this
    # file to know exactly which images were held out for THIS model.
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({name: [filenames[i] for i in idx] for name, idx in
                   (("train", train_idx), ("val", val_idx), ("test", test_idx))}, f, indent=2)

    def make_dataset(indices, image_transforms):
        dataset = TIRFDataset(config["image_dir"], config["mask_dir"], image_transforms)
        dataset.images = [filenames[i] for i in indices]
        return dataset

    train_dataset = make_dataset(train_idx, train_transform)
    val_dataset = make_dataset(val_idx, eval_transform) if val_idx else None
    test_dataset = make_dataset(test_idx, eval_transform)

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False) if test_dataset else None

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(config["classes"], device)
    if config["pretrained_weights"]:
        load_pretrained_weights(model, config["pretrained_weights"], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print(f"Learning rate: {config['lr']}")

    # --- train ---
    writer = SummaryWriter(os.path.join(out_dir, "logs"))
    best_state = train(model, train_loader, val_loader, optimizer, device, config["epochs"], writer)
    writer.close()

    torch.save(best_state, os.path.join(model_dir, 'best_model.pth'))
    print(f"Saved model weights to {os.path.join(model_dir, 'best_model.pth')}")

    # --- test prediction using the best model's weight ---
    if test_loader is not None:
        model.load_state_dict(best_state)
        save_test_predictions(model, test_loader, test_dataset, config["colormap"], predictions_dir, device)
        print(f"Saved test predictions to {predictions_dir}")
    else:
        print("Not enough image held-out for testing - skipping test predictions.")

if __name__ == "__main__":
    main()
