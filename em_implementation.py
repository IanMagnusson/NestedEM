"""
Implementations of the EM algorithm for a single data set or a combination of 2 data sets, where all data sets are
generated from a mixture of the same two gaussians.
"""
import numpy as np
import scipy.stats  # used just for performant gaussian pdf functions

MAX_STEPS = 100

def compute_EM(data):
    """
    A function to compute the EM algorithm for a single dataset from a mixture of two gaussians
    :param data: a numpy array of floats sampled from the gaussian mixture
    :return: a list of floats, mean estimates; a list of floats, standard deviation estimates; a float, alpha estimate
    """
    # initialize starting parameters
    mu = np.array([data.max(), data.min()])  # means, extrema chosen to avoid underflow from unlikely data
    sg = np.ones(2)  # standard deviations
    beta = .5  # weight of first component

    t = 0
    while t < MAX_STEPS:  # until max steps or no parameters change
        # Save to check convergence
        old = np.concatenate((mu, sg, [beta]))

        # get component PDFs based on old parameters
        N_0 = scipy.stats.norm(mu[0], sg[0])
        N_1 = scipy.stats.norm(mu[1], sg[1])

        # compute component priors from old parameters
        py_0 = (beta * N_0.pdf(data)) / (beta * N_0.pdf(data) + (1 - beta) * N_1.pdf(data))
        py_1 = 1 - py_0

        # update params
        beta = sum(py_0) / len(data)

        mu[0] = sum(py_0 * data) / sum(py_0)
        mu[1] = sum(py_1 * data) / sum(py_1)

        sg[0] = np.sqrt(sum(py_0 * np.square(data - mu[0])) / sum(py_0))
        sg[1] = np.sqrt(sum(py_1 * np.square(data - mu[1])) / sum(py_1))

        t += 1
        # if none of the parameters have changed, finish
        new = np.concatenate((mu, sg, [beta]))
        if (np.isclose((new - old).max(), 0.0)):
            break

    return mu, sg, beta


def compute_EM_two_datasets(data1, data2):
    """
    A function to compute the EM algorithm for two datasets from a mixture of the same two gaussians with dif weights
    :param data1: a numpy array of floats sampled from the gaussian mixture
    :param data2: a numpy array of floats sampled from the gaussian mixture
    :return: a list of floats, mean estimates; a list of floats, standard deviation estimates;
     a float, alpha estimate; a float, beta estimate
    """
    # initialize starting parameters
    mu = np.array([data1.max(), data1.min()])  # means, extrema chosen to avoid underflow from unlikely data
    sg = np.ones(2)  # standard deviations
    alpha = .5  # weight of first component
    beta = .5  # weight of first component

    t = 0
    while t < MAX_STEPS:  # until max steps or no parameters change
        # Save to check convergence
        old = np.concatenate((mu,sg,[alpha],[beta]))

        # get component PDFs based on old parameters
        N_0 = scipy.stats.norm(mu[0], sg[0])
        N_1 = scipy.stats.norm(mu[1], sg[1])

        # compute component priors from old parameters
        py_0_d1 = (alpha * N_0.pdf(data1)) / (alpha * N_0.pdf(data1) + (1 - alpha) * N_1.pdf(data1))
        py_1_d1 = 1 - py_0_d1

        py_0_d2 = (beta * N_0.pdf(data2)) / (beta * N_0.pdf(data2) + (1 - beta) * N_1.pdf(data2))
        py_1_d2 = 1 - py_0_d2

        # update params
        alpha = sum(py_0_d1) / len(data1)
        beta = sum(py_0_d2) / len(data2)

        mu[0] = (sum(py_0_d1 * data1) + sum(py_0_d2 * data2)) / (sum(py_0_d1) + sum(py_0_d2))
        mu[1] = (sum(py_1_d1 * data1) + sum(py_1_d2 * data2)) / (sum(py_1_d1) + sum(py_1_d2))

        sg[0] = np.sqrt((sum(py_0_d1 * np.square(data1 - mu[0])) + sum(py_0_d2 * np.square(data2 - mu[0])))
                        / (sum(py_0_d1) + sum(py_0_d2)))
        sg[1] = np.sqrt((sum(py_1_d1 * np.square(data1 - mu[1])) + sum(py_1_d2 * np.square(data2 - mu[1])))
                        / (sum(py_1_d1) + sum(py_1_d2)))
        t += 1

        # if none of the parameters have changed, finish
        new = np.concatenate((mu,sg,[alpha],[beta]))
        if(np.isclose((new - old).max(), 0.0)):
            break



    return mu, sg, alpha, beta
