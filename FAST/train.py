#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from torchvision.transforms.functional import crop
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import adjust_brightness

# Limit GPU memory usage
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

NUM_RUNS = 10  # K-Fold Cross Validation
NUM_EPOCHS = 100  # Epochs per fold

base_dir = "unetplusplus_tuning"
os.makedirs(base_dir, exist_ok=True)

# TODO: load the paths from config file or environment variable
alternate_cells = {
    "u2os": {"image_dir": "figures_fast/figure3/u2os_images", "mask_dir": "figures_fast/figure3/u2os_masks"},
    "llcpk1": {"image_dir": "figures_fast/figure3/llcpk1_images", "mask_dir": "figures_fast/figure3/llcpk1_masks"},
}

for run in range(NUM_RUNS):
    run_dir = os.path.join(base_dir, f"run_{run}")
    os.makedirs(run_dir, exist_ok=True)

    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    # TODO: define data augmentaions in seperate module
    class RandomCropPair:
        def __init__(self, crop_size, resize_to=None, p=0.5):
            """
            Args:
                crop_size (tuple): The size of the crop (height, width).
                resize_to (tuple): The size to resize the image and mask after cropping (height, width).
                p (float): Probability of applying the random crop.
            """
            self.crop_size = crop_size
            self.resize_to = resize_to
            self.p = p

        def __call__(self, image, mask):
            # Apply random crop with probability p
            if random.random() < self.p:
                i, j, h, w = self.get_params(image, self.crop_size)
                image = crop(image, i, j, h, w)
                mask = crop(mask, i, j, h, w)

            # Resize to a fixed size if specified
            if self.resize_to:
                image = resize(image, self.resize_to)
                mask = resize(mask, self.resize_to)

            return image, mask

        @staticmethod
        def get_params(image, crop_size):
            """Get parameters for `crop` for a random crop."""
            width, height = image.size
            crop_height, crop_width = crop_size

            if width < crop_width or height < crop_height:
                raise ValueError("Crop size must be smaller than image size.")

            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            return top, left, crop_height, crop_width

    random_crop = RandomCropPair(crop_size=(800, 800), resize_to=(1600, 1600), p=0.25)

    class ConfocalDataset(Dataset):
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
            mask_name = img_name.replace(".tif", ".png")
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")

            # Apply random crop
            if self.random_crop:
                image, mask = self.random_crop(image, mask)

            # Apply other transformations
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

        def __repr__(self):
            return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"

    class RandomBrightness:
        def __init__(self, brightness_factor_range=(0.5, 1.5), p=0.5):
            """
            Args:
                brightness_factor_range (tuple): Range of brightness factors to apply.
                p (float): Probability of applying the brightness adjustment.
            """
            self.brightness_factor_range = brightness_factor_range
            self.p = p

        def __call__(self, image):
            # Apply random brightness adjustment with probability p
            if random.random() < self.p:
                brightness_factor = random.uniform(*self.brightness_factor_range)
                image = adjust_brightness(image, brightness_factor)
            return image

        def __repr__(self):
            return f"{self.__class__.__name__}(brightness_factor_range={self.brightness_factor_range}, p={self.p})"

    # Define transformations for training, validation, and test datasets
    train_transform_image = transforms.Compose(
        [
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
            RandomBrightness(brightness_factor_range=(0.8, 1.2), p=0.5),
        ]
    )

    train_transform_mask = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_test_transform_image = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_test_transform_mask = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create separate datasets for training, validation, and testing
    train_dataset = ConfocalDataset(
        image_dir="figures_fast/figure2/confocal_images", mask_dir="figures_fast/figure2/confocal_masks", transforms=[train_transform_image, train_transform_mask], random_crop=random_crop
    )

    val_dataset = ConfocalDataset(
        image_dir="figures_fast/figure2/confocal_images",
        mask_dir="figures_fast/figure2/confocal_masks",
        transforms=[val_test_transform_image, val_test_transform_mask],
        random_crop=None,  # No random cropping for validation
    )

    test_dataset = ConfocalDataset(
        image_dir="figures_fast/figure2/confocal_images",
        mask_dir="figures_fast/figure2/confocal_masks",
        transforms=[val_test_transform_image, val_test_transform_mask],
        random_crop=None,  # No random cropping for testing
    )

    TRAIN_SIZE = 99
    VAL_SIZE = 20
    TEST_SIZE = 30

    all_indices = list(range(len(train_dataset.images)))
    random.shuffle(all_indices)

    train_indices = all_indices[:TRAIN_SIZE]
    val_indices = all_indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_indices = all_indices[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    train_dataset.images = [train_dataset.images[i] for i in train_indices]
    val_dataset.images = [val_dataset.images[i] for i in val_indices]
    test_dataset.images = [test_dataset.images[i] for i in test_indices]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

    classes = 5
    print(f"Train size: {len(train_loader)*3}, Val size: {len(val_loader)*3}, Test size: {len(test_loader)*3}")

    class_counts = Counter()

    for _, mask in train_dataset:
        mask = mask.long().squeeze(0)
        class_counts.update(mask.flatten().tolist())

    total_pixels = sum(class_counts.values())

    class_weights = {cls: total_pixels / count for cls, count in class_counts.items()}

    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    class_weights_list = [class_weights[i] for i in range(len(class_weights))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)

    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    # TODO: remove focal loss
    def focal_loss(pred, target, alpha=1, gamma=2):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        logpt = -torch.nn.functional.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** gamma) * logpt

        return focal_loss.mean()

    # TODO: remove focal loss anc ce loss since we only use dice loss
    def combined_loss(pred, target, class_weights_tensor, alpha=1.0, beta=0.2, gamma=0.4):
        dice = dice_loss(pred, target)
        # ce = nn.CrossEntropyLoss(weight=class_weights_tensor)(pred, target)
        # focal = focal_loss(pred, target)
        return alpha * dice  # + gamma * (focal) #+ beta * ce

    # TODO: move this to utils module
    def decode_segmentation_masks(mask, colormap, n_classes):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(0, n_classes):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    # Loading the Colormap
    colormap = np.array(
        [
            [0.0, 0.0, 0.0],  # Background - Black
            [0.0, 0.99, 0.0],  # Actin - Green
            [0.99, 0.0, 0.0],  # Focal Adhesions - Red
            [0.99, 0.99, 0.0],  # Lamellipodia - Yellow
            [0.0, 0.0, 0.99],  # Filopodia - Blue
        ]
    )

    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)

    model_name = "UnetPlusPlus"
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=classes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lowest_val_loss = float("inf")
    lowest_val_loss_abs = float("inf")
    best_model = None
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks, class_weights_tensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log training loss
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(images)
                loss = combined_loss(outputs, masks, class_weights_tensor)
                val_loss += loss.item()

        # Log validation loss
        writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            lowest_val_loss_abs = val_loss / len(val_loader)
            best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    writer.close()

    torch.save(best_model, os.path.join(model_dir, "best_model.pth"))

    # Save predictions
    predictions_dir = os.path.join(model_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    model.load_state_dict(best_model)
    model.eval()

    all_preds = []
    all_masks = []

    contour_counts_label_1 = []
    contour_counts_label_2 = []
    contour_counts_label_3 = []
    contour_counts_label_4 = []

    fractions_data = []
    contours_data = []

    # Define colors for each label
    colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 165, 0), 4: (0, 0, 255)}  # Green for Actin  # Red for Focal Adhesions  # Orange for Lamellipodia  # Blue for Filopodia

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks)

            batch_image_names = [test_dataset.images[i] for i in range(idx * test_loader.batch_size, min((idx + 1) * test_loader.batch_size, len(test_dataset)))]

            for i, img_name in enumerate(batch_image_names):
                input_image = images[i].cpu().numpy().transpose(1, 2, 0)
                decoded_mask = decode_segmentation_masks(masks[i], colormap, classes)
                decoded_pred = decode_segmentation_masks(preds[i], colormap, classes)

                input_image_rgb = np.repeat(input_image, 3, axis=2)
                input_image_rgb = np.clip(input_image_rgb, 0, 1)
                plt.imsave(os.path.join(predictions_dir, f"decoded_mask_{img_name}.png"), decoded_mask)
                plt.imsave(os.path.join(predictions_dir, f"decoded_pred_{img_name}.png"), decoded_pred)

                # Create overlay images
                gt_overlay = input_image_rgb.copy()
                pred_overlay = input_image_rgb.copy()

                for label, contour_counts in zip([1, 2, 3, 4], [contour_counts_label_1, contour_counts_label_2, contour_counts_label_3, contour_counts_label_4]):
                    # Get contours for the label
                    gt_contours, _ = cv2.findContours((masks[i] == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    kernel = np.ones((3, 3), np.uint8)
                    pred_contours, _ = cv2.findContours((preds[i] == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw contours on overlay images
                    cv2.drawContours(gt_overlay, gt_contours, -1, colors[label], 2)
                    cv2.drawContours(pred_overlay, pred_contours, -1, colors[label], 2)

                    for cont in gt_contours:
                        contours_data.append([img_name, "Ground Truth", label, cv2.contourArea(cont)])

                    for cont in pred_contours:
                        contours_data.append([img_name, "Predicted", label, cv2.contourArea(cont)])

                    # Define min_area for each label
                    min_area_dict = {1: 0, 2: 0, 3: 0, 4: 0}
                    min_area = min_area_dict.get(label, 20)  # Default to 20 if label is not in the dictionary

                    contour_counts.append((len([cont for cont in gt_contours if cv2.contourArea(cont) > min_area]), len([cont for cont in pred_contours if cv2.contourArea(cont) > min_area])))

                combined_overlay = np.concatenate((gt_overlay, pred_overlay), axis=1)
                # Normalize combined_overlay to ensure values are in the range [0, 1]
                combined_overlay = np.clip(combined_overlay, 0, 1)

                plt.imsave(os.path.join(predictions_dir, f"overlay_{img_name}.png"), combined_overlay)

                # Calculate fractions for ground truth and predicted masks
                for class_id in range(1, classes):
                    gt_fraction = np.sum(masks[i] == class_id) / np.sum(masks[i] != 0)
                    pred_fraction = np.sum(preds[i] == class_id) / np.sum(preds[i] != 0)
                    fractions_data.append([img_name, "Ground Truth", class_id, gt_fraction])
                    fractions_data.append([img_name, "Predicted", class_id, pred_fraction])

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    contours_df = pd.DataFrame(contours_data, columns=["Image", "Type", "Class", "Area"])
    contours_df.to_csv(os.path.join(model_dir, "contour_areas.csv"), index=False)

    fractions_df = pd.DataFrame(fractions_data, columns=["Image", "Type", "Class", "Fraction"])
    fractions_df.to_csv(os.path.join(model_dir, "class_fractions.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Class", y="Fraction", hue="Type", data=fractions_df)
    plt.title(f"Average Fraction of Each Class for {model_name}")
    plt.savefig(os.path.join(model_dir, "class_fraction_boxplot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))

    colors = {1: "green", 2: "red", 3: "orange", 4: "blue"}
    labels = {1: "Actin", 2: "Focal Adhesions", 3: "Lamellipodia", 4: "Filopodia"}

    for label in range(1, 5):
        gt_data = contours_df[(contours_df["Class"] == label) & (contours_df["Type"] == "Ground Truth")]
        pred_data = contours_df[(contours_df["Class"] == label) & (contours_df["Type"] == "Predicted")]

        gt_counts = gt_data.groupby("Image")["Area"].count().reset_index(name="Ground Truth Contours")
        pred_counts = pred_data.groupby("Image")["Area"].count().reset_index(name="Predicted Contours")

        merged_counts = pd.merge(gt_counts, pred_counts, on="Image", how="outer").fillna(0)

        sns.scatterplot(x="Ground Truth Contours", y="Predicted Contours", data=merged_counts, label=labels[label], color=colors[label])

        X = merged_counts["Ground Truth Contours"].values.reshape(-1, 1)
        y = merged_counts["Predicted Contours"].values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        plt.plot(merged_counts["Ground Truth Contours"], y_pred, color=colors[label], linestyle="--", linewidth=0.5, label=f"{labels[label]} Best fit line (R² = {r2:.2f})")

    # Add y = x line
    plt.plot([1, 200], [1, 200], color="black", linestyle="--", linewidth=1, label="y = x")

    # Set logarithmic scales
    plt.xscale("log")
    plt.yscale("log")

    # Set axis labels and title
    plt.xlabel("Number of detections in Ground Truth Mask (log scale)")
    plt.ylabel("Number of detections in Prediction Mask (log scale)")
    plt.title(f"Scatter Plot of Actin Subtypes for {model_name}")
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"contour_scatter_plot_log_{lowest_val_loss_abs}.png"))

    plt.close()

    print(f"Lowest validation loss: {lowest_val_loss_abs}")

    # ----------------------------------------------------------------------------------------------------------------
    # Process alternate cell types
    # ----------------------------------------------------------------------------------------------------------------

    for cell_type, paths in alternate_cells.items():
        print(f"Processing alternate cell type: {cell_type}")

        # Load dataset for the alternate cell type
        alt_dataset = ConfocalDataset(image_dir=paths["image_dir"], mask_dir=paths["mask_dir"], transforms=[val_test_transform_image, val_test_transform_mask], random_crop=None)

        alt_loader = DataLoader(alt_dataset, batch_size=3, shuffle=False)

        alt_contours_data = []
        alt_fractions_data = []

        with torch.no_grad():
            for idx, (images, masks) in enumerate(alt_loader):
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                masks = masks.cpu().numpy()

                batch_image_names = [alt_dataset.images[i] for i in range(idx * alt_loader.batch_size, min((idx + 1) * alt_loader.batch_size, len(alt_dataset)))]

                for i, img_name in enumerate(batch_image_names):
                    decoded_mask = decode_segmentation_masks(masks[i], colormap, classes)
                    decoded_pred = decode_segmentation_masks(preds[i], colormap, classes)

                    plt.imsave(os.path.join(predictions_dir, f"{cell_type}_decoded_mask_{img_name}.png"), decoded_mask)
                    plt.imsave(os.path.join(predictions_dir, f"{cell_type}_decoded_pred_{img_name}.png"), decoded_pred)

                    for label in range(1, classes):
                        # Get contours for the label
                        gt_contours, _ = cv2.findContours((masks[i] == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        pred_contours, _ = cv2.findContours((preds[i] == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for cont in gt_contours:
                            alt_contours_data.append([img_name, "Ground Truth", label, cv2.contourArea(cont)])

                        for cont in pred_contours:
                            alt_contours_data.append([img_name, "Predicted", label, cv2.contourArea(cont)])

                        # Calculate fractions for ground truth and predicted masks
                        gt_fraction = np.sum(masks[i] == label) / np.sum(masks[i] != 0)
                        pred_fraction = np.sum(preds[i] == label) / np.sum(preds[i] != 0)
                        alt_fractions_data.append([img_name, "Ground Truth", label, gt_fraction])
                        alt_fractions_data.append([img_name, "Predicted", label, pred_fraction])

        alt_contours_df = pd.DataFrame(alt_contours_data, columns=["Image", "Type", "Class", "Area"])
        alt_fractions_df = pd.DataFrame(alt_fractions_data, columns=["Image", "Type", "Class", "Fraction"])

        alt_model_dir = model_dir
        alt_contours_df.to_csv(os.path.join(alt_model_dir, f"{cell_type}_contour_areas.csv"), index=False)
        alt_fractions_df.to_csv(os.path.join(alt_model_dir, f"{cell_type}_class_fractions.csv"), index=False)

        # Plot box plot for fractions
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Class", y="Fraction", hue="Type", data=alt_fractions_df)
        plt.title(f"Average Fraction of Each Class for {cell_type}")
        plt.savefig(os.path.join(alt_model_dir, f"{cell_type}_class_fraction_boxplot.png"))
        plt.close()

        # Plot scatter plot for contours
        plt.figure(figsize=(10, 6))
        for label in range(1, 5):
            gt_data = alt_contours_df[(alt_contours_df["Class"] == label) & (alt_contours_df["Type"] == "Ground Truth")]
            pred_data = alt_contours_df[(alt_contours_df["Class"] == label) & (alt_contours_df["Type"] == "Predicted")]

            gt_counts = gt_data.groupby("Image")["Area"].count().reset_index(name="Ground Truth Contours")
            pred_counts = pred_data.groupby("Image")["Area"].count().reset_index(name="Predicted Contours")

            merged_counts = pd.merge(gt_counts, pred_counts, on="Image", how="outer").fillna(0)

            sns.scatterplot(x="Ground Truth Contours", y="Predicted Contours", data=merged_counts, label=labels[label], color=colors[label])

            X = merged_counts["Ground Truth Contours"].values.reshape(-1, 1)
            y = merged_counts["Predicted Contours"].values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            plt.plot(merged_counts["Ground Truth Contours"], y_pred, color=colors[label], linestyle="--", linewidth=0.5, label=f"{labels[label]} Best fit line (R² = {r2:.2f})")

        # Add y = x line
        plt.plot([1, 100], [1, 100], color="black", linestyle="--", linewidth=1, label="y = x")

        # Set logarithmic scales
        plt.xscale("log")
        plt.yscale("log")

        # Set axis labels and title
        plt.xlabel("Number of detections in Ground Truth Mask (log scale)")
        plt.ylabel("Number of detections in Prediction Mask (log scale)")
        plt.title(f"Scatter Plot of Actin Subtypes for {cell_type}")
        plt.legend()
        plt.savefig(os.path.join(alt_model_dir, f"{cell_type}_contour_scatter_plot_log.png"))
        plt.close()
