### Methods and constants related to the processing of signals
### Author: Florian Ramakers

from scipy.io import wavfile
import numpy as np
from librosa import cqt


""" The maximal value of a signed 16-bit integer. This value is used to normalize the 16-bit WAVE data to the float interval [-1.0, 1.0]. """
MAX_INT_16 = 32768

""" The sample rate used in all generated samples. """
SAMPLE_RATE = 44100

""" The lowest frequency that will be analysed (C2) """
MIN_FREQ = 65.41

""" The amount of bins per octave, used in the Constant Q-Transform. """
BINS_PER_OCTAVE = 24

""" The amount of bins to use in the Constant Q-Transform. """
N_BINS = 5 * BINS_PER_OCTAVE

def read_wav(path):
    """
    Read a WAVE (.wav) file. Currently only mono audio signals sampled at 44100 Hz are supported.
    :param path: The path to the audio file.
    :return: A Numpy array containing the audio data as Numpy floats.
    """
    _, data = wavfile.read(path)
    mono = np.array(data[:]).astype(np.float) / MAX_INT_16

    return mono

def generate_single(duration, specification):
    """
    Generate a signal that consists of a single note and that has the given duration.
    :param duration: The duration of the signal.
    :param specification: A tuple (s,f) where s represents the number of the string (zero based, starting from low E) and f the fret.
    :return: A Numpy array containing float data in the interval [-1.0, 1.0] that represents the specified signal and that has the specified duration.
    """
    data = read_wav('samples/{}_{}.wav'.format(specification[0], specification[1]))
    return data[:int(duration * SAMPLE_RATE)]

def generate_signal(duration, specification):
    """
    Generate a signal that consists of a multiple notes and that has the given duration.
    :param duration: The duration of the signal.
    :param specification: A list of tuples (s,f) where s represents the number of the string (zero based, starting from low E) and f the fret.
    :return: A Numpy array containing float data in the interval [-1.0, 1.0] that represents the specified signal and that has the specified duration.
    """
    data = np.zeros(int(duration * SAMPLE_RATE))
    for specification in specification:
        data += (1 / len(specification)) * generate_single(duration, specification)
    return data

def calc_constant_q_transform(signal):
    """
    Calculate the Constant Q-Transform of the given signal.
    :param signal: The signal to calculate the Constant Q-Transform of.
    :return: A numpy array containing the absolutes of the Constant Q-Transform of the given signal.
    """
    return np.abs(cqt(signal, sr=SAMPLE_RATE, fmin=MIN_FREQ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))