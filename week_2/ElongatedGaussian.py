import numpy as np


def elongated_gaussian(x_1: float, x_2: float):
    """
    elongated gaussian which we want to approach by sampling using different
    methods
    :param x_1:
    :param x_2:
    :return:
    """
    vec_x = np.array([x_1, x_2])
    a = np.array([[250.25, -249.75],
                  [-249.75, 250.25]])

    norm_const = 1 / np.sqrt((2 * np.pi) ** 2 * (1 / np.linalg.det(a)))
    return norm_const * np.exp(-0.5 * np.transpose(vec_x).dot(a.dot(vec_x)))

