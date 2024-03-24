import numpy as np
import tensorflow as tf

from settings import *


def save_model(model, version=1):
    model.save(f"{OCR_MODEL_BASENAME}v{version}.keras")
    model.save_weights(f"{OCR_MODEL_BASENAME}v{version}.weights.h5")
    print(f"Saved model {OCR_MODEL_BASENAME}v{version} to disk")


def load_model(version=1):
    # Load the model
    model = tf.keras.models.load_model(f"{OCR_MODEL_BASENAME}v{version}.keras")

    # Load weights into new model
    model.load_weights(f"{OCR_MODEL_BASENAME}v{version}.weights.h5")
    print(f"Loaded model {OCR_MODEL_BASENAME}v{version} from disk")

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def scale_input(inp):
    return inp / 255.0


def predict(model, imgs):
    if isinstance(imgs, np.ndarray):
        inputs = [scale_input(imgs)]
    else:
        print("Invalid input")
        return

    img_arr = np.asarray(inputs)
    img_arr = img_arr.reshape(img_arr.shape[0], *INPUT_SHAPE)

    res = model.predict(img_arr).reshape(len(COMBINED_CHARS), 1)

    results_dict = {}
    for i, c in enumerate(COMBINED_CHARS):
        results_dict[c] = res[i].astype(float)[0]

    return (COMBINED_CHARS[res.argmax()], sorted(results_dict.items(), key=lambda x: x[1], reverse=True))
