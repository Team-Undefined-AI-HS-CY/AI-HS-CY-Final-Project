import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import functools
from PIL import Image, ImageDraw
import math

from settings import *


def scale_to_size(img, verbose=False):
    scale_y = Y_SIZE / img.shape[0]
    scale_x = X_SIZE / img.shape[1]
    if verbose:
        print(f"Resizing {img.shape} to ({Y_SIZE},{X_SIZE}) with scale_x:{scale_x}, scale_y:{scale_y}")
    # Intercubic is slower but creates less blur overall
    return cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


# def clean_image(gray_img, threshold_area=0.3, verbose=False, invert=True):
#     if invert:
#         _, binary_img = cv2.threshold(gray_img, 0, BG_COLOR, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray_img, 0, BG_COLOR, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # _, binary_img = cv2.threshold(gray_img, 45, FG_COLOR, cv2.THRESH_BINARY)

#     # show_image(binary_img, "Binary Image")

#     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     if verbose:
#         img_copy = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
#         cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)
#         show_image(img_copy, "Contours drawn", cmap="bgr")

#     largest_contour_area = cv2.contourArea(contours[0])

#     # Remove small contours
#     for contour in contours:
#         contour_area = cv2.contourArea(contour)
#         # print(f"Contour area: {contour_area}, Largest contour area: {largest_contour_area}")
#         if contour_area < threshold_area * largest_contour_area:
#             cv2.drawContours(binary_img, [contour], -1, BG_COLOR, -1)
#         else:
#             if verbose:
#                 print(f"Keeping contour with area: {contour_area}")

#     return binary_img

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


# def dilate_image(img, verbose=False, invert=True):
#     blurred = cv2.GaussianBlur(img, (1, 1), 0)
#     thresh = cv2.adaptiveThreshold(blurred, BG_COLOR, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 15)

#     return thresh

def dilate_image(img, verbose=False, invert=True):
    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    if invert:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    else:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 15)

    return thresh



def prepare_img_for_training(img):
    img = scale_to_size(img)
    # img = dilate_image(img)
    # clean after dilate
    img = clean_image(img, invert=True)

    return img


def prepare_img_for_prediction(img):
    img = scale_to_size(img)
    # img = dilate_image(img, invert=False)
    img = clean_image(img, invert=True)

    return img


def pixelate_image(img):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (Y_SIZE, X_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def show_image(img, title="image", cmap="gray"):
    if cmap.lower() == "bgr":
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_images(images, cmap="gray", figsize=(5, 5)):
    # show bgr
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(images))
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        if not cmap:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")

    plt.tight_layout()
    plt.show()


def create_flow_chart(images, save=False, output_path="flow_chart.png"):
    # Convert images to PIL Image objects if they are NumPy arrays
    images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]
    # Calculate dimensions for the flow chart
    width = max(image.size[0] for image in images)
    height = sum(image.size[1] for image in images) + (len(images) - 1) * 50

    # Create a blank canvas for the flow chart
    flow_chart = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(flow_chart)

    # Initialize y-coordinate for placing images
    y_offset = 0

    # Paste images onto the flow chart and draw arrows between them
    increment = 50
    for image in images:
        x_offset = (width - image.size[0]) // 2
        flow_chart.paste(image, (x_offset, y_offset))
        if y_offset < height - image.size[1]:
            # make the arrow
            draw.line(
                [
                    (width // 2, y_offset + image.size[1]),
                    (width // 2 - increment, y_offset + image.size[1] + increment),
                ],
                fill="black",
                width=10,
            )
            draw.line(
                [
                    (width // 2, y_offset + image.size[1]),
                    (width // 2 + increment, y_offset + image.size[1] + increment),
                ],
                fill="black",
                width=10,
            )
        y_offset += image.size[1] + 50

    # Save or display the flow chart
    if save:
        flow_chart.save(output_path)
    else:
        flow_chart.show()

    return flow_chart


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


# def scale_bounding_boxes(boxes, image_width, image_height):

#     # Calculate average width and height of the bounding boxes
#     avg_width = np.mean([box[2] for box in boxes])
#     avg_height = np.mean([box[3] for box in boxes])
#     print(f"Average width: {avg_width}, Average height: {avg_height}")

#     # width_to_whole_ratio = avg_width / image_width
#     # height_to_whole_ratio = avg_height / image_height

#     # # decide on a scaling factor proportional to the ratios, the bigger the ratio the smaller the scaling factor
#     # K = 1 - (width_to_whole_ratio + height_to_whole_ratio) / 2
#     # avg_width = int(avg_width * K)
#     # avg_height = int(avg_height * K)

#     new_boxes = []

#     for box in boxes:
#         # top left x,y
#         x, y, w, h = box

#         w_ratio = w / avg_width
#         h_ratio = h / avg_height

#         new_w = avg_width * w_ratio
#         new_h = avg_height * h_ratio

#         new_x = max(0, x + (w - new_w) / 2)
#         new_y = max(0, y + (h - new_h) / 2)

#         # Ensure the new box stays within the image boundaries
#         # new_x = min(new_x, image_width - new_w)
#         # new_y = min(new_y, image_height - new_h)

#         # Calculate new top-left corner coordinates
#         new_x = max(0, x - (new_w - w) / 2)
#         new_y = max(0, y - (new_h - h) / 2)


#         new_x, new_y, new_w, new_h = int(new_x), int(new_y), int(new_w), int(new_h)

#         print(f"Old box: {box}, New box: {(new_x, new_y, new_w, new_h)}")

#         new_boxes.append((int(new_x), int(new_y), int(new_w), int(new_h)))

#     return new_boxes


def scale_bounding_boxes(boxes, image_width, image_height):

    new_boxes = []

    for box in boxes:
        x, y, w, h = box

        top_pad, bottom_pad, left_pad, right_pad = get_scaled_box_padding(box, boxes, image_width, image_height)

        new_x = max(0, x - left_pad)
        new_y = max(0, y - top_pad)
        new_w = w + left_pad + right_pad
        new_h = h + top_pad + bottom_pad

        new_x, new_y, new_w, new_h = int(new_x), int(new_y), int(new_w), int(new_h)

        print(f"Old box: {box}, New box: {(new_x, new_y, new_w, new_h)}")

        new_boxes.append((int(new_x), int(new_y), int(new_w), int(new_h)))

    return new_boxes


def get_scaled_box_padding(box, boxes, image_width, image_height):

    # Calculate average width and height of the bounding boxes
    avg_width = np.mean([box[2] for box in boxes])
    avg_height = np.mean([box[3] for box in boxes])

    width_ratio = avg_width / image_width
    height_ratio = avg_height / image_height

    x_ratio = Y_SIZE / X_SIZE
    y_ratio = 1  # X_SIZE / Y_SIZE

    scale_factor_x = BOUNDING_BOX_SCALE * (1 - width_ratio) * x_ratio
    scale_factor_y = BOUNDING_BOX_SCALE * (1 - height_ratio) * y_ratio

    avg_width = avg_width * scale_factor_x
    avg_height = avg_height * scale_factor_y

    x, y, w, h = box

    left_pad = (avg_width - w) // 2
    right_pad = (avg_width - w) // 2

    top_pad = (avg_height - h) // 2
    bottom_pad = (avg_height - h) // 2

    print(f"Old box: {box}, New box: {(x, y, w, h)}")
    print(f"Padding: {top_pad, left_pad, bottom_pad, right_pad}")

    return abs(int(top_pad)), abs(int(bottom_pad)), abs(int(left_pad)), abs(int(right_pad))


def get_components_mask(img, components):
    mask = np.zeros(img.shape, dtype="uint8")
    for label_mask in components:
        mask = cv2.add(mask, label_mask)

    return mask


def visualize_bounding_boxes(img, bounding_boxes, predictions):

    avg_width = np.mean([box[2] for box in bounding_boxes])
    avg_height = np.mean([box[3] for box in bounding_boxes])

    avg_area = avg_width * avg_height

    font_scale_factor = 70
    border_size_factor = 25

    font_scale = max(1, math.sqrt(avg_area) // font_scale_factor)
    border_size = max(1, math.sqrt(avg_area) // border_size_factor)

    img_copy = img.copy()
    for box, prediction in zip(bounding_boxes, predictions):
        x, y, w, h = box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), int(border_size))
        cv2.putText(
            img_copy,
            prediction,
            (x, int(y - font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            int(border_size),
        )
    return img_copy


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


def get_top_left_char(boxes):
    top_left = boxes[0]
    for box in boxes:
        if box[0] < top_left[0]:  # and box[1] < top_left[1]:
            top_left = box

    return top_left


def get_bottom_right_char(boxes, top_left, height):
    # must be bottom rightest character which does not fall more than h below top left
    bottom_right = top_left
    for box in boxes:
        if box[0] > bottom_right[0] and box[1] - top_left[1] < height:
            bottom_right = box

    return bottom_right


def get_gradient(c1, c2):
    print(f"Finding the gradient between {c1} and {c2}")
    gradient = (c2[1] - c1[1]) / (c2[0] - c1[0])

    return gradient


def get_angle_of_rotation(gradient):
    return np.arctan(gradient) * 180 / np.pi


def rotate_image(img, degrees):
    # Rotate the image
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))

    return rotated
