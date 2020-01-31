import numpy as np


def orientation(weights, grid):
    """
    Calculates the dot product of the provided weights with the xy-values of the
    grid.
    :param weights: n by 3 dimensional vector, where n is the amount of data
    points, 0 is the bias variable, 1 in the x direction and 2 in the y.
    :param grid: 2rr-dimensional tensor, with r the resolution of the grid.

    :return: rr-dimensional matrix with at each point a percentage of how
    many lines classify that xy-coordinate as + or - 1.
    """
    cnt = np.zeros(shape=(np.shape(grid[0])))
    print("Creating probability grid with resolution {}x{}."
          .format(len(grid[0]), len(grid[1])))
    print("|", end='')
    for i, w in enumerate(weights):
        if i / len(weights) * 100 % 5 == 0:
            print("█", end='')
        prod = w[0] + w[1] * grid[0] + w[2] * grid[1]
        prod[prod < 0] = 0
        cnt += np.sign(prod)
    print("|")

    # Normalize to a percentage
    cnt = cnt / cnt.max()
    return cnt
