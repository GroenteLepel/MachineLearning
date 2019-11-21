import numpy as np


def metropolis_hastings(p):
    """
    Method for sampling data from a distribution p
    :param p: distribution to sample using this method
    :return:
    """
    # copy- pasted from interwebs
    # ===========================
    x, y = 0., 0.
    samples = np.zeros(2)

    x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
    if (accept(p, normal_dist, np.array([x_star, y_star]), np.array([x, y]))):
        samples = np.array([x, y])

    return samples
    # ============================
    # end copy paste


def accept(p, q, sample_x, current_state):
    """
    Calculate the accaptence ratio between sample_x x' and current_state x
    according to the distributions p and q, and returns the proper parameter
    :param p: distribution to sample
    :param q: distribution picked by the metropolis_hastings method
    :param sample_x:
    :param current_state:
    :return:
    """
    a = p(sample_x) * q(current_state, sample_x) / \
        p(current_state) * q(sample_x, current_state)

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False
