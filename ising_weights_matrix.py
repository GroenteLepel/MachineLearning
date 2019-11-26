import numpy as np


def generate_matrix(n=100):
    # Generate a nxn matrix
    w = np.random.normal(size=(n, n))

    # Make it symmetric, and normalise
    w = w + w.transpose() / 2

    # Set diagonal to zero, no spin interactions with itself
    np.fill_diagonal(w, 0)

    return w