from week_2.p_star_distribution import p_star_distribution, objective_function
import numpy as np
from week_2.metropolis_hastings import metropolis_hastings
from plottebakkers import plot_mcmc
import functools
import matplotlib.pyplot as plt


def circle(x):
    return x[0] ** 2 + x[1] ** 2


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
                              n_points=1100)

remove_first = 1000

m_values = np.zeros(len(samples[remove_first:]))
for i, s in enumerate(samples[remove_first:]):
    m_values[i] = objective_function(s, x, t, 0.01)

# %% plotting
w = samples[remove_first:]

fig, ax = plt.subplots(1, 1)
plot_mcmc.plot_bayesian_solution(ax, w, x, t)
fig.show()

xmin, xmax = 1, 10
ymin, ymax = 1, 8
nsteps = complex(0, len(w))
grid = np.mgrid[xmin:xmax:nsteps, ymin:ymax:nsteps]

xrange = grid[0, :, 0]
yrange = grid[1, 0, :]

xgrid = grid[0, :, :]

# lines = plot_mcmc.line(grid, w)
ops = plot_mcmc.outer_points(grid, w)

p = ops[:, :2].transpose()
q = ops[:, 2:].transpose()

lines = plot_mcmc.line(grid, w)
pg = plot_mcmc.orientation(p, q, grid)

plt.plot(xrange, lines)
plt.scatter(grid[0], grid[1], c=pg)
plt.ylim(yrange.min(), yrange.max())
plt.show()
