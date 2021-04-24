# -*- coding: utf-8 -*-
"""DeepLearning-HW02"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.layers import Conv2D, Input
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import *
from keras.regularizers import l2


def preprocess_image(path, new_shape):
    """
    Prepares an image
    :param path:
    :param new_shape:
    :return:
    """
    image = Image.open(path)
    new_image = image.resize((105, 105))
    return np.asarray(new_image).reshape(new_shape)


def build_pairs(dataset_path, instrcut_path, new_shape):
    """
    Builds the class pairs for training
    :param dataset_path:
    :param instrcut_path:
    :param new_shape:
    :return:
    """
    inputs = []
    labels = []
    with open(instrcut_path, 'r') as file:
        number_true_labels = int(file.readline())
        line_index = 1

        # For the matching images
        while line_index <= number_true_labels:
            last_line = file.readline()
            name, pic_1, pic_2 = last_line.rstrip('\n').split("\t")
            pic1_path = dataset_path + name + "/" + name + "_" + pic_1.rjust(4, '0') + ".jpg"
            pic2_path = dataset_path + name + "/" + name + "_" + pic_2.rjust(4, '0') + ".jpg"
            inputs.append([preprocess_image(item, new_shape) for item in [pic1_path, pic2_path]])
            labels.append(1)

            line_index += 1

        # For the non-matching images
        while True:
            last_line = file.readline()
            if not last_line:
                break
            name_1, pic_1, name_2, pic_2 = last_line.rstrip('\n').split("\t")
            pic1_path = dataset_path + name_1 + "/" + name_1 + "_" + pic_1.rjust(4, '0') + ".jpg"
            pic2_path = dataset_path + name_2 + "/" + name_2 + "_" + pic_2.rjust(4, '0') + ".jpg"
            inputs.append([preprocess_image(item, new_shape) for item in [pic1_path, pic2_path]])
            labels.append(0)

    return np.array(inputs), np.array(labels)


def prepare_siamese_one_hot_model(input_shape):
    """
    Prepare and get the model
    :param input_shape:
    :return:
    """

    def initialize_weights(obj_shape, **kwargs):
        return np.random.normal(loc=0.0, scale=1e-2, size=obj_shape)

    def initialize_bias(obj_shape, **kwargs):
        return np.random.normal(loc=0.5, scale=1e-2, size=obj_shape)

    left_input = Input(input_shape)
    right_input = Input(input_shape)
    siamese_model = Sequential([
        Conv2D(64,
               (10, 10),
               activation='relu',
               input_shape=input_shape,
               kernel_initializer=initialize_weights,
               kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128,
               (7, 7),
               activation='relu',
               kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias,
               kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128,
               (4, 4),
               activation='relu',
               kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias,
               kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(256,
               (4, 4),
               activation='relu',
               kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias,
               kernel_regularizer=l2(2e-4)),
        Flatten(),
        Dense(4096,
              activation='sigmoid',
              kernel_regularizer=l2(1e-3),
              kernel_initializer=initialize_weights, bias_initializer=initialize_bias)])

    L1_distance_function = Lambda(lambda tens: K.abs(tens[0] - tens[1]))
    L1_distance = L1_distance_function([siamese_model(left_input), siamese_model(right_input)])

    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


def get_batch(data, index, batch_size):
    """
    Get the next batch
    :param data:
    :param index:
    :param batch_size:
    :return:
    """
    return np.array(data[index: index + batch_size])


def create_plot(title, xlabel, ylabel, until, x, y, legend):
    """
    Create a plot
    :param title:
    :param xlabel:
    :param ylabel:
    :param until:
    :param x:
    :param y:
    :param legend:
    :return:
    """
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    for item in y:
        plt.plot(x[:min(len(item), until)], item[:min(len(item), until)])
    plt.legend(legend)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(curr_dir, "content/lfw2/lfw2/")
    train_path = os.path.join(curr_dir, "content/pairsDevTrain.txt")
    test_path = os.path.join(curr_dir, "content/pairsDevTest.txt")
    shape = (105, 105, 1)
    num_epochs = 50
    check_test_acc = 1
    index_in_data = 0
    validation_split = 0.2

    train_inputs, train_labels = build_pairs(dataset_path, train_path, shape)
    test_inputs, test_labels = build_pairs(dataset_path, test_path, shape)

    # preprocess the data shapes to fit the model.fit
    train_set = np.split(train_inputs, 2, axis=1)
    train_set = [np.squeeze(item, 1) for item in train_set]

    test_set = np.split(test_inputs, 2, axis=1)
    test_set = [np.squeeze(item, 1) for item in test_set]

    all_test_losses = []
    all_test_accs = []
    all_train_losses = []
    all_train_accs = []
    # model.summary()

    for lr in [0.00005, 0.00006]:
        for batch_size in [32, 64]:

            model = prepare_siamese_one_hot_model(shape)
            optimizer = Adam(lr=lr)
            accuracy_func = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
            model.compile(loss="binary_crossentropy", optimizer=optimizer,
                          metrics=[accuracy_func])

            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            epoch_indexes = []

            first_time = True
            for iteration in range(num_epochs):
                # evaluate model on test set
                if iteration % check_test_acc == 0:
                    print("\nEvaluating...")
                    test_loss, test_acc = model.evaluate(x=test_set, y=test_labels)
                    print("test loss: {}, test acc: {}".format(test_loss, test_acc))
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    if not first_time:
                        if ((test_losses[-2] - test_losses[-1]) / test_losses[-2]) < 0.01:
                            print("Early Stopping after {} epochs".format(iteration))
                            break
                    else:
                        first_time = False
                    epoch_indexes.append(iteration)
                    print("Done Evaluating\n")

                print("epoch number {} with lr {}, batch_size {}".format(iteration, lr, batch_size))

                history = model.fit(x=train_set, y=train_labels, batch_size=batch_size)
                train_losses.append(history.history['loss'][0])
                train_accs.append(history.history['binary_accuracy'][0])

            all_test_losses.append(test_losses)
            all_test_accs.append(test_accs)
            all_train_losses.append(train_losses)
            all_train_accs.append(train_accs)

    create_plot("Test Accuracy", "Epochs", "Percent", 15, epoch_indexes, all_test_accs,
                ["lr = 0.00005, batch_size = 32", "lr = 0.00005, batch_size = 64", "lr = 0.00006, batch_size = 32",
                 "lr = 0.00006, batch_size = 64"])
    create_plot("Test Loss", "Epochs", "Loss", 15, epoch_indexes, all_test_losses,
                ["lr = 0.00005, batch_size = 32", "lr = 0.00005, batch_size = 64", "lr = 0.00006, batch_size = 32",
                 "lr = 0.00006, batch_size = 64"])
