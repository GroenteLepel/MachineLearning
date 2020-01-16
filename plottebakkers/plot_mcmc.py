import matplotlib.pyplot as plt
import numpy as np

DATAFOLDER = "../data/"


def line(xgrid_transposed, w):
    return -(w[:, 1] * xgrid_transposed + w[:, 0]) / w[:, 2]


def propability_grid(weights):
    xmin, xmax = 1, 10
    ymin, ymax = 1, 8
    nsteps = complex(0, len(weights))
    grid = np.mgrid[xmin:xmax:nsteps, ymin:ymax:nsteps]

    xrange = grid[0, :, 0]
    yrange = grid[1, 0, :]

    #generate the xgrid (tranposed by default)
    xgrid = grid[0, :, :]

    lines = line(xgrid, weights)


def plot_w_vs_iteration(axes, weights):
    axes.set_title(r'$w$ vs iteration')
    axes.plot(weights[:, 0], label=r'$w_1$')
    axes.plot(weights[:, 1], label=r'$w_2$')
    axes.plot(weights[:, 2], label=r'$w_3$')
    axes.legend()


def plot_m_vs_iteration(axes, m_values):
    axes.set_title(r'$M$ vs iteration')
    axes.plot(m_values)


def plot_spread(axes, weights):
    axes.set_title(r'$(w_1, w_2)$ sampled after burn-in')
    axes.scatter(weights[:, 1], weights[:, 2], marker='.')
    axes.set_xlabel(r'$w_2$')
    axes.set_ylabel(r'$w_1$')


def plot_bayesian_solution(axes, weights, data, labels):
    samples = np.linspace(2, 9, len(weights))
    axes.set_title('Bayesian solution')
    axes.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2],
                 marker='v', linewidths=3, c='black')
    axes.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2],
                 marker='o', linewidths=3, c='black')
    # axes.plot(samples,
    #           line(samples, weights[-10]),
    #           c="black")
    axes.set_ylim(1.9, 7.1)


def plotfig(weights, m_values, data, labels,
            show: bool = True, fname: str = ""):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    plot_w_vs_iteration(ax[0, 0], weights)

    plot_m_vs_iteration(ax[0, 1], m_values)

    plot_spread(ax[1, 0], weights)

    plot_bayesian_solution(ax[1, 1], weights, data, labels)

    if show:
        fig.show()
    else:
        if fname == "":
            fname = "mcmc_full.png"
        fig.savefig("{}{}".format(DATAFOLDER, fname))
