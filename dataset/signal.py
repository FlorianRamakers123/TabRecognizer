from scipy.io import wavfile
import numpy as np

""" The maximal value of a signed 16-bit integer. This value is used to normalize the 16-bit WAVE data to the float interval [-1.0, 1.0]. """
MAX_INT_16 = 32768

""" The sample rate used in all generated samples. """
SAMPLE_RATE = 44100

def read_wav(path):
    """
    Read a WAVE (.wav) file. Currently only mono audio signals sampled at 44100 Hz are supported.
    :param path: The path to the audio file.
    :return: A Numpy array containing the audio data as Numpy floats.
    """
    _, data = wavfile.read(path)
    mono = np.array(data[:]).astype(np.float) / MAX_INT_16

    return mono

def generate_single(duration, note):
    """
    Generate a signal that consists of a single note and that has the given duration.
    :param duration: The duration of the signal.
    :param note: The note that should be present in the signal (i.e. E4, Bb3). For flats and sharps the most common name is chosen (i.e. use 'Eb' and not 'D#').
    :return: A Numpy array containing float data in the interval [-1.0, 1.0] that represents the specified note and that has the specified duration.
    """
    data = read_wav('../samples/{}.wav'.format(note))
    return data[:int(duration * SAMPLE_RATE)]

def generate_signal(duration, notes):
    """
    Generate a signal that consists of a multiple notes and that has the given duration.
    :param duration: The duration of the signal.
    :param notes: A list containing the notes that should be present in the signal (i.e. [E4, Bb3]). For flats and sharps the most common name is chosen (i.e. use 'Eb' and not 'D#').
    :return: A Numpy array containing float data in the interval [-1.0, 1.0] that represents the specified notes and that has the specified duration.
    """
    data = np.zeros(int(duration * SAMPLE_RATE))
    for note in notes:
        data += (1 / len(notes)) * generate_single(duration, note)
    return data