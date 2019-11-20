def metropolis_hastings(p):
    """
    Method for sampling data from a distribution p
    :param p: distribution to sample using this method
    :return:
    """


def acceptance(p, q, sample_x, current_state):
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
        return sample_x
    else:
        return current_state
