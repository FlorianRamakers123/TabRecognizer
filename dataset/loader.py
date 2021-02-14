### Python file related to the loading of the datasets.
### Currently only 22-fretted 6-string guitars in standard E-tuning is supported.
### Author: Florian Ramakers

from dataset.generator import STRING_COUNT, TEST_SET_PATH, TRAINING_SET_PATH, IMAGE_WIDTH, IMAGE_HEIGHT
import glob
import tensorflow as tf
import numpy as np

def load_dataset(string_index, base_path):
    """
    Load the dataset for the given string from the given location.
    :param string_index: The index of the string to retrieve the data for.
    :param base_path: The base path of the dataset.
    :return: a tuple x, y where x is the input data and y the associated labels.
    """
    x = []
    y = []
    for img_path in glob.glob(base_path + '*.png'):
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
        x.append(tf.keras.preprocessing.image.img_to_array(img))

    f = open(base_path + 'labels.txt')
    labels = f.readlines()
    for label in labels[string_index::STRING_COUNT]:
        y.append(np.array(list(map(int, label.strip()))))

    return np.array(x), np.array(y)

def load_train_data(string_index):
    """
    Load the training set for the given string.
    :param string_index: The index of the string to retrieve the data for.
    :return: a tuple x, y where x is the input data of the training set and y the associated labels.
    """
    return load_dataset(string_index, TRAINING_SET_PATH)

def load_test_data(string_index):
    """
    Load the test set for the given string.
    :param string_index: The index of the string to retrieve the data for.
    :return: a tuple x, y where x is the input data of the test set and y the associated labels.
    """
    return load_dataset(string_index, TEST_SET_PATH)
