import numpy as np
import cv2
from tqdm.auto import tqdm
import os
from utils.images import show_image, pixelate_image, prepare_img_for_training, augment_img
from settings import *

currend_dir = os.path.dirname(__file__)
dataset_name = os.path.basename(currend_dir)

def load_dataset(include=COMBINED_CHARS):
    X_data = []
    Y_data = []

    for i, c in tqdm(enumerate(COMBINED_CHARS), desc=f"Loading {dataset_name}", total=len(COMBINED_CHARS), leave=True):
        if c not in include:
            continue
        res_matrix = np.zeros(len(COMBINED_CHARS))
        res_matrix[i] = 1 # change the according character position in the result matrix sucessful percentage of 1 (this would be the ideal output)
        # print(res_matrix)
        char_folder = f"{currend_dir}/class_{c}"
        if not os.path.exists(char_folder):
            print(f"Folder {c} does not exist")
            continue
        for img_path in tqdm(os.listdir(char_folder), desc=f"Loading {c}", leave=True):
            img = cv2.imread(f"{char_folder}/{img_path}", cv2.IMREAD_GRAYSCALE)
            img = prepare_img_for_training(img)
            # show_image(img)
            # break

            X_augmented = augment_img(img)

            X_data.extend(X_augmented)
            Y_data.extend([res_matrix] * len(X_augmented))



    X_data = np.asarray(X_data)
    Y_data = np.asarray(Y_data)

    return X_data, Y_data
