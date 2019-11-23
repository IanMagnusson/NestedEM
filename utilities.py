"""
Some utility functions for computing the experiment results
"""

import numpy as np

def get_sample_means(data):
    """
    Calculate the sample means for each column
    :param data: a list of lists
    :return: a list of floats, the means
    """
    return (np.array(data).sum(axis=0) / len(data)).tolist()

def get_sample_variance(data):
    """
    Calculate the sample variance for each column
    :param data: a list of lists
    :return: a list of floats, the variances
    """
    means = get_sample_means(data)
    return (np.square(np.subtract(data, means)).sum(axis=0) / len(data)).tolist()
