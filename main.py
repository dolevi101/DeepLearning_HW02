import numpy as np
import tensorflow as tf
from tensorflow import keras


if __name__ == "__main__":
    dataset_path = "lfw2" #the relative path inside my project folder
    # i'm following this guide: https://keras.io/getting_started/intro_to_keras_for_engineers/
    dataset = keras.preprocessing.image_dataset_from_directory(
        dataset_path, batch_size=64, image_size=(250, 250))

    for data, labels in dataset:
        print(data.shape)  # (64, 200, 200, 3)
        print(data.dtype)  # float32
        print(labels.shape)  # (64,)
        print(labels.dtype)  # int32