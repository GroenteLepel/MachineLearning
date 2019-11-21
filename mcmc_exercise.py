from p_star_distribution import p_star_distribution
import numpy as np
from metropolis_hastings import metropolis_hastings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

w = np.arange(-10, 10, 0.5)

w_samples = metropolis_hastings(p_star_distribution, x, t, n_points=5000)


# %%
def line(x, w0, w1, w2):
    return -(w1 * x + w0) / w2


points = np.linspace(2, 9, 1000)
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

ax[0].plot(w_samples[:, 0], label='w1')
ax[0].plot(w_samples[:, 1], label='w2')
ax[0].plot(w_samples[:, 2], label='w3')

ax[0].legend()

ax[1].scatter(w_samples[:, 1], w_samples[:, 2])

ax[2].scatter(x[:, 1], x[:, 2], c=t)
# ax[2].plot(points,
#            line(points, w_samples[-1, 0], w_samples[-1, 1], w_samples[-1, 2]),
#            lw=5)

fig.show()
