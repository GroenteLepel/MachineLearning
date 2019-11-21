from p_star_distribution import p_star_distribution
import numpy as np

w1 = np.random.normal(size=3)
w2 = np.random.normal(size=3)

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

a = p_star_distribution(w1, x, t)
b = p_star_distribution(w2, x, t)

fraction = a / bg
