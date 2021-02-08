import numpy as np

def normalize(x):
    return x / np.max(np.abs(np.min(x)), np.max(x))
