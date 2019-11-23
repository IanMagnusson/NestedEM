import numpy as np

class DataGenerator:
    """A gaussian mixture random data generator"""
    def __init__(self, num_components, component_means, component_standard_deviations):
        """
        Make a data generator with some number of specified gaussian components
        :param num_components: an int
        :param component_means: a list of floats, in order of the corresponding component
        :param component_standard_deviations:  a list of floats, in order of the corresponding component
        """
        self.num_comp = num_components
        self.mus = component_means
        self.sigmas = component_standard_deviations

    def get_data(self, weights, size):
        """
        Get an array of samples of size length from components by specified weights
        :param weights: a list of floats summing to 1
        :param size: an int
        :return: a numpy array of floats containing the sampled data
        """
        # a sequence of choices of which component to sample
        sample_idxes = np.random.choice(self.num_comp, size=size, p=weights)

        # build array of samples based on sequence of indices
        return np.fromiter((np.random.normal(self.mus[i], self.sigmas[i]) for i in sample_idxes), dtype=np.float_)
