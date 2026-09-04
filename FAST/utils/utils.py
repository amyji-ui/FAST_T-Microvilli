#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import numpy as np
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from scipy.io import loadmat
from typing import List, Tuple, Union

import cv2
import tifffile

import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
from tensorflow import keras
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
from keras import layers
from tensorflow.python.keras.layers import Activation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with open("config.json", "r") as config_file:
    config = json.load(config_file)

IMAGE_SIZE = config["IMAGE_SIZE"]
NUM_CLASSES = config["NUM_CLASSES"]
DATA_DIR = config["DATA_DIR"]
NUM_TRAIN_IMAGES = config["NUM_TRAIN_IMAGES"]
NUM_VAL_IMAGES = config["NUM_VAL_IMAGES"]
EPOCHS = config["EPOCHS"]
LEARNING_RATE = config["LEARNING_RATE"]


train_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


# Loading the Colormap
colormap = np.array([[0.0 , 0.0, 0.0], # Background - Black
                     [0.0, 0.99, 0.0], # Cell Body - Green
                     [0.99, 0.0, 0.0]]) # Protrusions - Red

colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def get_percentages(mask: np.ndarray) -> List[str]:
    """
    Calculate the percentage of each class in the mask.

    Args:
        mask (ndarray): Segmentation mask.

    Returns:
        list: List of strings representing the class name and percentage for each class.
    """
    class_counts = np.bincount(mask.astype(np.int64).flatten())
    total_pixels = np.sum(class_counts[1:])

    # Exclude background
    class_percentages = class_counts[1:] / total_pixels * 100  

    classmap = {1: "cortex", 2: "fibers", 3: "filo", 4: "lamellar", 5: "lamfilo"}

    # Add legend text to the mask
    legend_text = [
        f"{classmap[i]}: {percentage:.2f}%"
        for i, percentage in enumerate(class_percentages, start=1)
    ]

    return legend_text


def decode_segmentation_masks(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Decode the segmentation mask into RGB color channels.

    Args:
        mask (ndarray): Segmentation mask.
        n_classes (int): Number of classes.

    Returns:
        ndarray: RGB image representing the colored mask.
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    pseudo_mask = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
        pseudo_mask[idx] = l

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image: Image.Image, colored_mask: np.ndarray) -> np.ndarray:
    """
    Create an overlay of the image and colored mask.

    Args:
        image (PIL.Image.Image): Input image.
        colored_mask (ndarray): RGB image representing the colored mask.

    Returns:
        ndarray: Overlay image.
    """
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.45, colored_mask, 0.55, 0)
    return overlay


def plot_samples_matplotlib(display_list: List[Union[np.ndarray, Image.Image]], figsize: Tuple[int, int] = (5, 3), mask: np.ndarray = None) -> Image.Image:
    """
    Plot a list of images using matplotlib.

    Args:
        display_list (list): List of images to be displayed.
        figsize (tuple, optional): Figure size. Defaults to (5, 3).
        mask (ndarray, optional): Segmentation mask. Defaults to None.

    Returns:
        PIL.Image.Image: Image of the plotted samples.
    """
    titles = ["Image", "Prediction Mask", "Overlay"]
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    legend_text = get_percentages(mask)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
            # Adding legend to the prediction mask
            if i == 1:
                for j, text in enumerate(legend_text, start=1):
                    axes[i].text(
                        0.05,
                        1.0 - 0.05 * j,
                        text,
                        color=list(colormap[j]/100),
                        fontsize=10,
                        transform=axes[i].transAxes,
                        va="top",
                        ha="left",
                    )
        else:
            axes[i].imshow(display_list[i])
        axes[i].title.set_text(titles[i])
    
    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    return img


def create_pdf(image_folder: str, output_pdf: str) -> None:
    """
    Create a PDF document based on saved images.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_pdf (str): Path to the output PDF file.
    """
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
    c = canvas.Canvas(output_pdf, pagesize=letter)

    for image_path in image_files:

        _ = Image.open(image_path)

        # Get the size of the image (assuming it's in portrait orientation)
        width, height = letter[::-1]
        width /= 1.25
        height /= 1.25

        c.drawInlineImage(image_path, 0, 0, width, height)
        
        c.setFont("Helvetica", 18)
        title = f"{os.path.dirname(image_path).split('/')[-1]}: {os.path.basename(image_path)}"
        c.drawCentredString(width / 2, height - 20, title)

        c.showPage()

    c.save()


def convert_to_png():
    # Define the source and target directories
    source_dir = 'blebs_master/cropped'
    target_dir = 'blebs_master/pngs'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # List all tiff files in the source directory
    tif_files = [f for f in os.listdir(source_dir) if f.endswith('.tif')]

    # Convert each tiff file to png and save in the target directory
    for tif_file in tif_files:
        # Read the tiff mask
        mask = tifffile.imread(os.path.join(source_dir, tif_file))
        # Define the png file name (change extension to .png)
        png_file = tif_file.rsplit('.', 1)[0] + '.png'
        # Save the mask as png
        plt.imsave(os.path.join(target_dir, png_file), mask, cmap='gray')

def create_masks():
    # read tif files
    cell_body_files = os.listdir('data/cell_body')
    protrusion_files = sorted(os.listdir('data/protrusions'))

    for protrusion_file in protrusion_files:
        cell_body = tifffile.imread('data/cell_body/' + protrusion_file)
        protrusion = tifffile.imread('data/protrusions/' + protrusion_file)
        
        # create mask such that cell body is 1 and protrusion is 2
        mask = np.zeros_like(cell_body)
        
        mask[cell_body == 0] = 1
        mask[protrusion == 0] = 2

        # save mask
        tifffile.imsave('data/masks/' + protrusion_file, mask.astype(np.uint8))

def augment_image_and_mask(image, mask):
    # Combine image and mask to apply the same transformations
    combined = tf.concat([image, mask], axis=-1)
    
    # Apply random transformations
    combined = tf_image.random_flip_left_right(combined)
    combined = tf_image.random_flip_up_down(combined)
    combined = tf_image.random_crop(combined, size=[int(IMAGE_SIZE * 0.9), int(IMAGE_SIZE * 0.9), 4])
    combined = tf_image.resize(combined, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Split the combined image and mask
    image = combined[:, :, :3]
    mask = combined[:, :, 3:]
    
    # Apply additional augmentations to the image only
    image = tf_image.random_brightness(image, max_delta=0.1)
    image = tf_image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf_image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf_image.random_hue(image, max_delta=0.1)
    image = tf_image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
    
    return image, mask

def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
    image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])    
    return image


def load_data(image_path, mask_path, augment=False):
    image = read_image(image_path)
    mask = read_image(mask_path, mask=True)
    if augment:
        image, mask = augment_image_and_mask(image, mask)
    return image, mask

def data_generator(image_list, mask_list, augment=False):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(lambda x, y: load_data(x, y, augment), num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def refine_boundaries(predictions, num_classes=6):
    """
    Refine the boundaries of the predictions using morphological operations.
    
    :param predictions: The predicted mask (H, W) with class values
    :param num_classes: The number of classes
    :return: Refined mask (H, W) with class values
    """
    refined_predictions = np.zeros_like(predictions)
    
    for i in range(num_classes):
        # Convert predictions to binary mask for each class
        binary_mask = (predictions == i).astype(np.uint8)
        
        # Skip if the class is not present in the predictions
        if np.sum(binary_mask) == 0:
            continue
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        # Combine the refined mask back into the refined_predictions
        refined_predictions[refined_mask == 1] = i
    
    return refined_predictions


def infer(model, image_tensor, threshold=None, refine=False):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0), verbose=0)
    predictions = np.squeeze(predictions)
    if threshold and np.max(predictions) > np.min(predictions):
        predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
        predictions = np.where(predictions > threshold, predictions, 0)
    predictions = np.argmax(predictions, axis=2)
    if refine:
        return refine_boundaries(predictions, num_classes=5)
    return predictions


def plot_predictions(images_list, mask_list, colormap, model, head=False, dest_dir=f'testing/predictions/', display_percentage=False):
    for i, image_file in enumerate(images_list):
        if head and i > 0:
            break
        image_tensor = read_image(image_file)
        
        if mask_list is not None:
            mask = tf_io.read_file(mask_list[i]) # read_image(mask_list[i])
            mask = tf_image.decode_png(mask, channels=1)
            mask.set_shape([None, None, 1])
            mask = tf_image.resize(images=mask, size=[IMAGE_SIZE, IMAGE_SIZE])
            mask = np.squeeze(mask, axis=-1)
        
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        legend_text = False
        if display_percentage:
            legend_text = get_percentages(prediction_mask)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, len(colormap))
        overlay = get_overlay(image_tensor, prediction_colormap)
        os.makedirs(dest_dir, exist_ok=True)
        if mask_list is not None:
            
            mask_colormap = decode_segmentation_masks(mask, colormap, len(colormap))
            plot_samples_matplotlib(
                [image_tensor, prediction_colormap, mask_colormap], os.path.basename(image_file), figsize=(18, 14), dest_dir=dest_dir, display_percentage=legend_text
            )
        else:
            plot_samples_matplotlib(
                [image_tensor, prediction_colormap], os.path.basename(image_file), figsize=(18, 14), dest_dir=dest_dir, display_percentage=legend_text
            )

        
def display(display_list):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def analyze_blob(contour, conversion_factor=0.108, length=1, breadth=1):
    area_pixels = cv2.contourArea(contour) * (length * breadth)
    perimeter_pixels = cv2.arcLength(contour, True) * ((length + breadth) / 2)

    area_microns = area_pixels * (conversion_factor ** 2)
    perimeter_microns = perimeter_pixels * conversion_factor

    circularity = 0
    if perimeter_microns:
        circularity = 4 * np.pi * area_microns / (perimeter_microns ** 2)
        
    ellipse = cv2.fitEllipse(contour)
    major_diameter_pixels, minor_diameter_pixels = ellipse[1]
    major_diameter_microns = major_diameter_pixels * conversion_factor * ((length + breadth) / 2)
    minor_diameter_microns = minor_diameter_pixels * conversion_factor * ((length + breadth) / 2)
    if minor_diameter_microns > major_diameter_microns:
        major_diameter_microns, minor_diameter_microns = minor_diameter_microns, major_diameter_microns

    return {
      "area": area_microns,
      "circularity": circularity,
      "major_diameter": major_diameter_microns,
      "minor_diameter": minor_diameter_microns
    }

def label2_iou(y_true, y_pred):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)


    y_true_label2 = tf.equal(y_true, 2)
    y_pred_label2 = tf.equal(y_pred, 2)

    intersection = tf.reduce_sum(tf.cast(y_true_label2 & y_pred_label2, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true_label2 | y_pred_label2, tf.float32))

    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou