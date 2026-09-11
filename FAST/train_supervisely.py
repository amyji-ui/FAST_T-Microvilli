#!/usr/bin/env python
# coding: utf-8
"""
Train the FAST UNet++ segmentation model on a Supervisely-labeled dataset
that has been converted with convert_supervisely_annotations.py.

This is a trimmed-down version of train.py meant for getting a first model
off a small, growing set of manually labeled images: it does a single
train/val/test run (not the 10-run K-fold sweep) with split sizes that
scale to however many images you actually have, and it drops the
hardcoded alternate-cell-type (u2os/llcpk1) evaluation block from train.py,
since those datasets won't exist for a fresh project.

Usage (reads image/mask dirs, pretrained-weights, epochs, etc. from config.yaml):
    python train_supervisely.py --config config.yaml

All parameters can also be passed/overridden on the command line, e.g.:
    python train_supervisely.py \
        --config config.yaml \
        --image-dir "../Images/Labeled_converted/images" \
        --mask-dir "../Images/Labeled_converted/masks" \
        --pretrained-weights fast_model.pth \
        --epochs 100 \
        --out-dir supervisely_run

With very few labeled images (say, 10), expect the model to overfit; treat
this as a sanity check that the pipeline works end-to-end, and re-run once
you've labeled more images in Supervisely and re-exported/re-converted them.

fast_model.pth was trained with 5 output classes (background + Actin +
Focal Adhesion + Lamellipodia + Filopodia). This adaptation drops Filopodia
(classes=4), so when --pretrained-weights is given, everything except the
final segmentation_head layer (shape [5,...] vs [4,...]) is loaded from the
checkpoint; the segmentation head starts randomly initialized and has to
relearn its last layer from your labeled data.
"""

import argparse
import copy
import os
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import adjust_brightness, crop, resize

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class RandomCropPair:
    def __init__(self, crop_size, resize_to=None, p=0.5):
        self.crop_size = crop_size
        self.resize_to = resize_to
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            i, j, h, w = self.get_params(image, self.crop_size)
            image = crop(image, i, j, h, w)
            mask = crop(mask, i, j, h, w)
        if self.resize_to:
            image = resize(image, self.resize_to)
            mask = resize(mask, self.resize_to)
        return image, mask

    @staticmethod
    def get_params(image, crop_size):
        width, height = image.size
        crop_height, crop_width = crop_size
        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than image size.")
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return top, left, crop_height, crop_width


class TIRFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, random_crop=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.random_crop = random_crop
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.random_crop:
            image, mask = self.random_crop(image, mask)

        if self.transforms:
            image = self.transforms[0](image)
            mask = self.transforms[1](mask)

        mask = (mask * 255).long()
        return image, mask


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor


class RandomBrightness:
    def __init__(self, brightness_factor_range=(0.5, 1.5), p=0.5):
        self.brightness_factor_range = brightness_factor_range
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            image = adjust_brightness(image, brightness_factor)
        return image


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    return np.stack([r, g, b], axis=2)


# Fallback, used if config.yaml has no `colormap` key. Keys are mask pixel
# values (class ids); should match convert_supervisely_annotations.py's
# colormap and class-name-to-index.
DEFAULT_COLORMAP = {
    0: [0, 0, 0],  # Background
    1: [0, 255, 0],  # Actin
    2: [255, 0, 0],  # Focal Adhesions
    3: [255, 255, 0],  # Lamellipodia
}


def colormap_dict_to_array(colormap: dict) -> np.ndarray:
    max_idx = max(int(k) for k in colormap)
    arr = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for k, v in colormap.items():
        arr[int(k)] = v
    return arr


def split_sizes(n, train_frac=0.7, val_frac=0.15):
    """Scale train/val/test counts to however many images are available,
    guaranteeing at least 1 image in val/test once n >= 3."""
    train_n = max(1, round(n * train_frac))
    val_n = max(1, round(n * val_frac)) if n >= 3 else 0
    test_n = max(0, n - train_n - val_n)
    if test_n == 0 and n >= 3:
        # borrow one from train so test isn't empty
        train_n -= 1
        test_n = 1
    return train_n, val_n, test_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml; CLI flags below override its values")
    parser.add_argument("--image-dir", default=None, help="Overrides config's converted-dir/images")
    parser.add_argument("--mask-dir", default=None, help="Overrides config's converted-dir/masks")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None, help="Number of segmentation classes incl. background (background+Actin+Focal Adhesion+Lamellipodia; no Filopodia)")
    parser.add_argument("--pretrained-weights", default=None, help="Path to fast_model.pth (or another checkpoint) to fine-tune from. Layers whose shape doesn't match (e.g. segmentation_head, if --classes differs from the checkpoint) are left randomly initialized.")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to 1e-4 when fine-tuning from --pretrained-weights, else 1e-3.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    args.image_dir = args.image_dir or config.get("image-dir") or (os.path.join(config["converted-dir"], "images") if config.get("converted-dir") else None)
    args.mask_dir = args.mask_dir or config.get("mask-dir") or (os.path.join(config["converted-dir"], "masks") if config.get("converted-dir") else None)
    if not args.image_dir or not args.mask_dir:
        parser.error("image/mask dirs are required: set converted-dir (or image-dir/mask-dir) in --config, or pass --image-dir/--mask-dir")

    args.out_dir = args.out_dir or config.get("out-dir") or "supervisely_run"
    args.epochs = args.epochs if args.epochs is not None else config.get("epochs", 100)
    args.batch_size = args.batch_size if args.batch_size is not None else config.get("batch-size", 2)
    args.classes = args.classes if args.classes is not None else config.get("classes", 4)
    args.pretrained_weights = args.pretrained_weights or config.get("pretrained-weights")
    args.lr = args.lr if args.lr is not None else config.get("lr")
    if args.lr is None:
        args.lr = 1e-4 if args.pretrained_weights else 1e-3
    args.seed = args.seed if args.seed is not None else config.get("seed", 42)
    colormap = colormap_dict_to_array(config.get("colormap", DEFAULT_COLORMAP))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, "model")
    predictions_dir = os.path.join(model_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out_dir, "logs"))

    random_crop = RandomCropPair(crop_size=(400, 400), resize_to=None, p=0.0)  # disabled by default for small images

    train_transform_image = transforms.Compose(
        [
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
            RandomBrightness(brightness_factor_range=(0.8, 1.2), p=0.5),
        ]
    )
    train_transform_mask = transforms.Compose([transforms.ToTensor()])
    val_test_transform_image = transforms.Compose([transforms.ToTensor()])
    val_test_transform_mask = transforms.Compose([transforms.ToTensor()])

    base_dataset = TIRFDataset(args.image_dir, args.mask_dir)
    n_images = len(base_dataset.images)
    if n_images < 2:
        raise SystemExit(f"Need at least 2 labeled images to train, found {n_images} in {args.image_dir}")

    train_n, val_n, test_n = split_sizes(n_images)
    print(f"Found {n_images} labeled images -> train={train_n}, val={val_n}, test={test_n}")

    all_indices = list(range(n_images))
    random.shuffle(all_indices)
    train_indices = all_indices[:train_n]
    val_indices = all_indices[train_n : train_n + val_n]
    test_indices = all_indices[train_n + val_n : train_n + val_n + test_n]

    all_filenames = base_dataset.images

    def make_dataset(transform_image, transform_mask, rc, indices):
        ds = TIRFDataset(args.image_dir, args.mask_dir, transforms=[transform_image, transform_mask], random_crop=rc)
        ds.images = [all_filenames[i] for i in indices]
        return ds

    train_dataset = make_dataset(train_transform_image, train_transform_mask, random_crop, train_indices)
    val_dataset = make_dataset(val_test_transform_image, val_test_transform_mask, None, val_indices) if val_indices else None
    test_dataset = make_dataset(val_test_transform_image, val_test_transform_mask, None, test_indices) if test_indices else None

    batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), shuffle=False) if test_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=args.classes,
    ).to(device)

    if args.pretrained_weights:
        pretrained_sd = torch.load(args.pretrained_weights, map_location=device)
        model_sd = model.state_dict()
        compatible, skipped = {}, []
        for k, v in pretrained_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)
        model_sd.update(compatible)
        model.load_state_dict(model_sd)
        print(f"Loaded {len(compatible)}/{len(pretrained_sd)} tensors from {args.pretrained_weights}")
        if skipped:
            print(f"  Left at random init (shape mismatch): {skipped}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Learning rate: {args.lr}")

    lowest_val_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)

        val_loss = None
        if val_loader:
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device).long().squeeze(1)
                    outputs = model(images)
                    running_val += dice_loss(outputs, masks).item()
            val_loss = running_val / len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
        else:
            # No val set (too few images) - just keep the latest weights.
            best_model = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            msg = f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f", Val Loss: {val_loss:.4f}"
            print(msg)

    writer.close()
    torch.save(best_model, os.path.join(model_dir, "best_model.pth"))
    print(f"Saved model weights to {os.path.join(model_dir, 'best_model.pth')}")

    model.load_state_dict(best_model)
    model.eval()

    if test_loader:
        with torch.no_grad():
            for idx, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()

                batch_names = [
                    test_dataset.images[i] for i in range(idx * test_loader.batch_size, min((idx + 1) * test_loader.batch_size, len(test_dataset)))
                ]
                for i, img_name in enumerate(batch_names):
                    decoded_mask = decode_segmentation_masks(masks_np[i], colormap, args.classes)
                    decoded_pred = decode_segmentation_masks(preds[i], colormap, args.classes)
                    plt.imsave(os.path.join(predictions_dir, f"decoded_mask_{img_name}"), decoded_mask)
                    plt.imsave(os.path.join(predictions_dir, f"decoded_pred_{img_name}"), decoded_pred)
        print(f"Saved test predictions to {predictions_dir}")
    else:
        print("No held-out test images (too few labeled images) - skipping test predictions.")


if __name__ == "__main__":
    main()
