import numpy as np
import week_2.p_star_distribution as p_star
from week_2.metropolis_hastings import metropolis_hastings
import week_2.hamilton as hamilton
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


def test_longeving_parameters():
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))

    step_sizes = np.array([0.01, 0.05, 0.1, 0.25])
    n_leaps = np.array([2, 5, 10, 20])
    for i, eps in enumerate(step_sizes):
        for j, lf in enumerate(n_leaps):
            hamilton_samples = hamilton.sample(4000, x, t,
                                               epsilon=eps, leap_frog_steps=lf)
            log = r"$\epsilon$ = {0:2.2f}, $n$ = {1:d}, $r_r$ = {2:3.2f}" \
                .format(eps, lf, hamilton_samples[3])
            plot_mcmc.plot_spread(ax[i][j], hamilton_samples[0], title=log)

    fig.tight_layout()
    fig.savefig("longeving_param.png", dpi=fig.dpi)


def final_test():
    n_samples = 8000
    mh_samples, a = metropolis_hastings(
        functools.partial(p_star.p_star_distribution, x, t), x, t,
        n_points=n_samples)

    plot_mcmc.plotfig(mh_samples, x, t, a)

    hamilton_samples = hamilton.sample(n_samples, x, t)
    plot_mcmc.plotfig(hamilton_samples[0], x, t)



# %% plotting
# samples, a = metropolis_hastings(
#     functools.partial(p_star_distribution, x, t), x, t,
#     n_points=8000)

# hamilton_samples = hamilton.sample(4000, x, t)

final_test()

# plot_mcmc.pl