from week_2.p_star_distribution import p_star_distribution, objective_function
import numpy as np
from week_2.metropolis_hastings import metropolis_hastings
import matplotlib.pyplot as plt
import functools


def circle(x):
    return x[0] ** 2 + x[1] ** 2


def line(x, w0, w1, w2):
    return -(w1 * x + w0) / w2


np.random.seed(2)

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

samples = metropolis_hastings(functools.partial(p_star_distribution, x, t), x,
                              t,
                              n_points=5000)

remove_first = 1000

m_values = np.zeros(len(samples[remove_first:]))
for i, s in enumerate(samples[remove_first:]):
    m_values[i] = objective_function(s, x, t, 0.01)

# %% plotting
w = samples[remove_first:]
x_samples = np.linspace(2, 9, len(w))
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].set_title(r'$w$ vs iteration')
ax[0, 0].plot(w[:, 0], label=r'$w_1$')
ax[0, 0].plot(w[:, 1], label=r'$w_2$')
ax[0, 0].plot(w[:, 2], label=r'$w_3$')
ax[0, 0].legend()

ax[0, 1].set_title(r'$M$ vs iteration')
ax[0, 1].plot(m_values)

ax[1, 0].set_title(r'$(w_1, w_2)$ sampled after burn-in')
ax[1, 0].scatter(w[:, 1], w[:, 2],
                 marker='.')
ax[1, 0].set_xlabel(r'$w_2$')
ax[1, 0].set_ylabel(r'$w_1$')

ax[1, 1].set_title('Bayesian solution')
ax[1, 1].scatter(x[t == 0][:, 1], x[t == 0][:, 2],
                 marker='+', linewidths=5, c='b')
ax[1, 1].scatter(x[t == 1][:, 1], x[t == 1][:, 2],
                 marker='o', linewidths=3, c='r')
ax[1, 1].plot(x_samples, line(x_samples, w[-10, 0], w[-10, 1], w[-10, 2]),
                 marker=',')
ax[1, 1].set_ylim(1.9, 7.1)

plt.show()
