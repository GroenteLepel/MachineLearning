import numpy as np


def generate_matrix(n=100, frustrated=True):
    # Generate a nxn matrix
    w = np.random.normal(size=(n, n))

    # Make it symmetric, and normalise
    w += w.transpose()
    w /= 2

    # Set diagonal to zero, no spin interactions with itself
    np.fill_diagonal(w, 0)

    if not frustrated:
        w[w < 0] *= -1

    return w
