from p_star_distribution import p_star_distribution
import numpy as np
from metropolis_hastings import metropolis_hastings
import seaborn as sns
import matplotlib.pyplot as plt
import functools


def circle(x):
    return x[0] ** 2 + x[1] ** 2


x = np.array([
    [1, 2, 3],
    [1, 3, 2],
    [1, 3, 6],
    [1, 5.5, 4.5],
    [1, 5, 3],
    [1, 7, 4],
    [1, 5, 6],
    [1, 8, 6],
    [1, 9.5, 5],
    [1, 9, 7]
])

t = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# samples = metropolis_hastings(circle, n_dims=2, n_points=10000)

samples = metropolis_hastings(functools.partial(p_star_distribution, x, t),
                                n_points=100000)

sns.jointplot(samples[:, 0], samples[:, 1])
plt.show()

plt.plot(samples[:, 0])
plt.plot(samples[:, 1])
plt.plot(samples[:, 2])

plt.show()
