import numpy as np


def compute_scale_down(input_size, output_size):
    """compute scale down factor of neural network, given input and output size"""
    return output_size[0] / input_size[0]


def prob_true(p):
    """return True with probability p"""
    return np.random.random() < p
