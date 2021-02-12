import numpy as np

def normalize(x):
    return x / np.max(np.abs(np.min(x)), np.max(x))

def scale(x, min_x=0.0, max_x=1.0):
    x_scd = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_scd * (max_x - min_x) + min_x
