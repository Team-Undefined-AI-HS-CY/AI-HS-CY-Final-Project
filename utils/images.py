import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import functools

from settings import *


def scale_to_size(img, verbose=False):
    scale_y = Y_SIZE / img.shape[0]
    scale_x = X_SIZE / img.shape[1]
    if verbose:
        print(f"Resizing {img.shape} to ({Y_SIZE},{X_SIZE}) with scale_x:{scale_x}, scale_y:{scale_y}")
    # Intercubic is slower but creates less blur overall
    return cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


def clean_image(gray_img, threshold_area=0.3, verbose=False, invert=True):
    # Threshold the image to create a binary image
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the area of the largest contour
    largest_contour_area = cv2.contourArea(contours[0])

    # Remove small contours
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < threshold_area * largest_contour_area:
            cv2.drawContours(binary_img, [contour], -1, 0, -1)

    # Invert the binary image to get the cleaned image
    if invert:
        cleaned_img = cv2.bitwise_not(binary_img)
    else:
        cleaned_img = binary_img

    return cleaned_img


# Sort the bounding boxes from left to right, top to bottom
# sort by Y first, and then sort by X if Ys are similar
def compare_rects(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]


def dilate_image(img, verbose=False, invert=True):
    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    if invert:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    else:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 15)

    return thresh


def prepare_img_for_training(img):
    img = scale_to_size(img)
    img = dilate_image(img)
    # clean after dilate
    img = clean_image(img)

    return img


def prepare_img_for_prediction(img):
    img = scale_to_size(img)
    img = dilate_image(img, invert=False)
    img = clean_image(img, invert=False)

    return img


def pixelate_image(img):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (Y_SIZE, X_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def show_image(img, title="image", cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_images(images, cmap="gray", background_color="yellow"):
    fig = plt.figure(figsize=(3, 5))
    # add yelow background
    fig.patch.set_facecolor("yellow")
    gs = gridspec.GridSpec(1, len(images))
    for i, image in enumerate(images):
        fig.add_subplot(gs[0, i])
        plt.imshow(image, cmap=cmap)
        plt.axis("off")
    plt.show()


def label_components(image):
    _, labels = cv2.connectedComponents(image)

    return labels


def filter_components(labels, lower, upper):
    components = []
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(labels.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        if lower < num_pixels < upper:
            components.append(label_mask)
    return components


def get_bounding_boxes(img):
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(bounding_boxes, key=functools.cmp_to_key(compare_rects))

def get_components_mask(img, components):
    mask = np.zeros(img.shape, dtype="uint8")
    for label_mask in components:
        mask = cv2.add(mask, label_mask)

    return mask

def visualize_bounding_boxes(img, bounding_boxes, predictions):
    for box, prediction in zip(bounding_boxes, predictions):
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.putText(img, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

def augment_img(img):
    augmented = []

    # Augment the data
    ## Blur the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # show_image(blurred)
    augmented.append(blurred)

    ## Pixelate the image
    # pixelated = pixelate_image(img)
    # show_image(pixelated)
    # augmented.append(pixelated)

    return augmented
