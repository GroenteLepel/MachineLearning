from week_2.p_star_distribution import p_star_distribution, objective_function
import numpy as np
from week_2.metropolis_hastings import metropolis_hastings
from plottebakkers import plot_mcmc
import functools
import matplotlib.pyplot as plt

DATAFOLDER = "./data/"

def circle(x):
    return x[0] ** 2 + x[1] ** 2


# np.random.seed(2)

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


# %% plotting
spreads = [10, 1, 0.1, 0.01]

fig, ax = plt.subplots(1, len(spreads), figsize=(len(spreads)*3, 2.5))

for i in range(len(spreads)):
    samples, a = metropolis_hastings(
        functools.partial(p_star_distribution, x, t), x, t,
        n_points=8000, spread=spreads[i])
    travel_length, dump = plot_mcmc.determine_travel(a)
    travel = samples[:travel_length]
    weights = samples[travel_length:]
    plot_mcmc.plot_spread(ax[i], travel, weights)
    ax[i].set_title(r"$\sigma = {}$".format(spreads[i]))

fig.tight_layout()
filename = "spread_influence.png"
fig.savefig("{}{}".format(DATAFOLDER, filename))
