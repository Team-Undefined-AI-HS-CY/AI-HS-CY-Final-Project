from numba import cuda
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from settings import *

def clean_session():
    device = cuda.get_current_device()
    device.reset()

def plot_model(history, version=1, save=False):
    fig = plt.figure(figsize=(14,5))
    grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    fig.add_subplot(grid[0])
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    fig.add_subplot(grid[1])
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.autoscale()
    plt.show()

    # save the plot
    if save:
        fig.savefig(f"{OCR_MODELS_DIR}/v{version}/graph.png")

def scale_input(inp):
    inp = inp.astype('float32')
    inp /= 255.0
    return inp

def scale_dataset(X_train, X_test, batch_size=32):
    num_train_samples = len(X_train)
    num_test_samples = len(X_test)

    for i in range(0, num_train_samples, batch_size):
        X_train[i:i+batch_size] = scale_input(X_train[i:i+batch_size])

    for i in range(0, num_test_samples, batch_size):
        X_test[i:i+batch_size] = scale_input(X_test[i:i+batch_size])

    return X_train, X_test

# def scale_dataset(X_train, X_test):
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')

#     X_train = scale_input(X_train)
#     X_test = scale_input(X_test)

#     return X_train, X_test

def shape_dataset(X_train, X_test, input_shape):
    X_train = X_train.reshape(X_train.shape[0], *input_shape)
    X_test = X_test.reshape(X_test.shape[0], *input_shape)

    return X_train, X_test

def split_dataset(X_data, Y_data, test_percent, random_state, shuffle):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=test_percent, random_state=random_state, shuffle=shuffle)

    return X_train, X_test, Y_train, Y_test

def print_dataset_info(X_data, X_train, Y_train, X_test, Y_test):
    print(f"Data samples: {X_data.shape[0]}")
    print(f"X_train shape: {X_train.shape[0]}")
    print(f"Y_train shape: {Y_train.shape[0]}")
    print(f"X_test shape: {X_test.shape[0]}")
    print(f"Y_test shape: {Y_test.shape[0]}")
