### Python file related to the generation of the datasets.
### Currently only 22-fretted 6-string guitars in standard E-tuning is supported.
### Author: Florian Ramakers


import random as rnd
import numpy as np
from skimage.io import imsave
from dataset.signal import generate_signal, calc_constant_q_transform
from util.data import scale

""" The path for storing the train data. """
TRAINING_SET_PATH = 'data/train/'

""" The path for storing the test data. """
TEST_SET_PATH = 'data/test/'

""" The size of the training set. """
TRAINING_SET_SIZE = 10

""" The size of the test set. """
TEST_SET_SIZE = 10

""" The width of the input images (set by function 'imsave' implicitly). """
IMAGE_WIDTH = 44

""" The height of the input images (set by function 'imsave' implicitly). """
IMAGE_HEIGHT = 120


""" The complete guitar neck and its corresponding nodes. """
# Fret:               0     1     2      3     4     5      6      7     8     9     10     11    12    13     14    15     16    17     18    19    20     21    22
GUITAR = np.array([ 'E2', 'F2', 'F#2', 'G2', 'Ab2', 'A2', 'Bb2', 'B2', 'C3', 'C#3', 'D3', 'Eb3', 'E3', 'F3', 'F#3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3', 'C4', 'C#4', 'D4',
                    'A2', 'Bb2', 'B2', 'C3', 'C#3', 'D3', 'Eb3', 'E3', 'F3', 'F#3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3', 'C4', 'C#4', 'D4', 'Eb4', 'E4', 'F4', 'F#4', 'G4',
                    'D3', 'Eb3', 'E3', 'F3', 'F#3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3', 'C4', 'C#4', 'D4', 'Eb4', 'E4', 'F4', 'F#4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 'C5',
                    'G3', 'Ab3', 'A3', 'Bb3', 'B3', 'C4', 'C#4', 'D4', 'Eb4', 'E4', 'F4', 'F#4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 'C5', 'C#5', 'D5', 'Eb5', 'E5', 'F5',
                    'B3', 'C4', 'C#4', 'D4', 'Eb4', 'E4', 'F4', 'F#4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 'C5', 'C#5', 'D5', 'Eb5', 'E5', 'F5',  'F#5', 'G5', 'Ab5', 'A5',
                    'E4', 'F4', 'F#4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 'C5', 'C#5', 'D5', 'Eb5', 'E5', 'F5',  'F#5', 'G5', 'Ab5', 'A5', 'Bb5', 'B5', 'C6', 'C#6', 'D6']).reshape(6, 23)

""" The maximal fret that is supported. """
MAX_FRET = 22

""" The amount of strings on the guitar. """
STRING_COUNT = 6

""" The maximum distance between frets on two different strings. """
MAX_CHORD_DIST = 4

""" The duration of the samples in seconds. """
SAMPLE_DURATION = 0.5


def generate_random_specification():
    """
    Generate a complete random specification with the following constraints:
        - The distance between two frets on two different strings will always be smaller than MAX_CHORD_DIST.
        - A string is only associated to one fret.
    :return: A list of tuples (s,f) with s the zero-based index of the string and f the fret for that string.
    """
    specification = []
    k = rnd.randrange(1, STRING_COUNT)
    strings = rnd.sample(range(STRING_COUNT), k=k)
    available_frets = range(MAX_FRET + 1)
    for string in strings:
        fret = rnd.choice(available_frets)
        specification.append((string, fret))
        available_frets = list(filter(lambda f: f == 0 or all(d == 0 or abs(f - d) < MAX_CHORD_DIST for (_, d) in specification), available_frets))

    return specification

def create_spectrogram(signal):
    """
    Create the image data for the spectrogram of the specified signal.
    :param signal: The signal to calculate the spectrogram for.
    :return: The black and white image data, specified in a np.uint8 array, for the spectrogram of the specified signal.
    """
    cqt = calc_constant_q_transform(signal)
    return 255 - scale(cqt, 0, 255).astype(np.uint8)

def generate_example():
    """
    Generate a random labeled example.
    The label consists of a boolean vector
    :return: A tuple (signal, sample) where signal contains the data of an audio signal and where sample is a
    """
    specification = generate_random_specification()
    outputs = [np.zeros(MAX_FRET + 1, dtype=bool) for _ in range(STRING_COUNT)]
    signal = generate_signal(SAMPLE_DURATION, specification)
    for s,f in specification:
        outputs[s][f] = True

    return signal, outputs

def generate_dataset(n, base_path):
    """
    Generate a dataset of the given length and store it in the given path.
    :param n: The desired size of the dataset.
    :param base_path: The path for storing the dataset.
    """
    f = open(base_path + "labels.txt", 'a')
    for i in range(n):
        signal, outputs = generate_example()
        for label in outputs:
            f.write("".join(label.astype(int).astype(str)) + '\n')
        img = create_spectrogram(signal)
        imsave(base_path + "{}.png".format(i), img)
    f.close()

def create_input_data():
    """
    Create the training set and test set and store them on the disk.
    """
    generate_dataset(TRAINING_SET_SIZE, TRAINING_SET_PATH)
    generate_dataset(TEST_SET_SIZE, TEST_SET_PATH)


def get_input_shape():
    """
    Get the input shape of the input data.
    :return: A tuple indicating the input size of the input data.
    """
    return IMAGE_HEIGHT, IMAGE_WIDTH, 1

def get_output_shape():
    """
    Get the output shape of the input data.
    :return: The output size of the input data.
    """
    return MAX_FRET + 1
