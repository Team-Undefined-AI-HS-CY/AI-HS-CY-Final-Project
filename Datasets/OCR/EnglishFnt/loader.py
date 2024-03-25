import numpy as np
import cv2
from tqdm.auto import tqdm
import os
from utils.images import show_image, pixelate_image, prepare_img_for_training
from settings import *

currend_dir = os.path.dirname(__file__)
dataset_name = os.path.basename(currend_dir)


def load_dataset(include_chars=COMBINED_CHARS, include_fonts=list(range(1, 1016))):
    X_data = []
    Y_data = []

    for i, c in tqdm(enumerate(COMBINED_CHARS), desc=f"Loading {dataset_name}", total=len(COMBINED_CHARS), leave=True):
        if c not in include_chars:
            # print(f"Skipping {c} {i}")
            continue
        res_matrix = np.zeros(len(COMBINED_CHARS))
        res_matrix[i] = (
            1  # change the according character position in the result matrix sucessful percentage of 1 (this would be the ideal output)
        )
        # print(res_matrix)
        char_num = str(i+1).rjust(3, "0")
        char_folder = f"{currend_dir}/English/Fnt/Sample{char_num}"
        if not os.path.exists(char_folder):
            print(f"Folder for {c} does not exist")
            continue
        for img_num in tqdm(include_fonts, desc=f"Loading {c}", leave=True):
            img_path = f"img{char_num}-{str(img_num).rjust(5, '0')}.png"
            # print(f"Reading {char_folder}/{img_path}")
            img = cv2.imread(f"{char_folder}/{img_path}", cv2.IMREAD_GRAYSCALE)
            # crop to character
            w, h = img.shape
            img = img[w // 10 : 9 * w // 10, h // 10 : 9 * h // 10]
            img = prepare_img_for_training(img)
            # show_image(img)
            # break
            X_data.append(img)
            Y_data.append(res_matrix)

            # Augment the data
            ## Blur the image
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            # show_image(blurred)
            # break
            X_data.append(blurred)
            Y_data.append(res_matrix)

            ## Pixelate the image
            pixelated = pixelate_image(img)
            # show_image(pixelated)
            # break
            X_data.append(pixelated)
            Y_data.append(res_matrix)

            # # Flip the image
            # flipped = np.fliplr(img)
            # X_data.append(flipped)
            # Y_data.append(res_matrix)

            # # Flip the pixelated image
            # flipped_pixelated = np.fliplr(pixelated)
            # X_data.append(flipped_pixelated)
            # Y_data.append(res_matrix)

    X_data = np.asarray(X_data)
    Y_data = np.asarray(Y_data)

    return X_data, Y_data
