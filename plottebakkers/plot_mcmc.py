import matplotlib.pyplot as plt
import numpy as np
import week_2.p_star_distribution as pstar

DATAFOLDER = "../data/"


def line(grid, weights):
    # generate the xgrid (tranposed by default)
    xgrid_transposed = grid[0, :, :]
    return -(weights[:, 1] * xgrid_transposed + weights[:, 0]) / weights[:, 2]


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
    print("|", end='')
    for i, w in enumerate(weights):
        if i / len(weights) * 100 % 5 == 0:
            print("█", end='')
        prod = w[0] + w[1] * grid[0] + w[2] * grid[1]
        prod[prod < 0] = 0
        cnt += np.sign(prod)
    print("|")

    # Normalize to a percentage
    cnt = cnt / len(weights[0])
    return cnt


def plot_w_vs_iteration(axes, weights):
    axes.set_title(r'$w$ vs iteration')
    axes.plot(weights[:, 0], label=r'$w_1$')
    axes.plot(weights[:, 1], label=r'$w_2$')
    axes.plot(weights[:, 2], label=r'$w_3$')
    axes.legend()


def plot_m_vs_iteration(axes, m_values):
    axes.set_title(r'$M$ vs iteration')
    axes.plot(m_values)


def plot_spread(axes, travel, weights):
    axes.set_title(r'$(w_1, w_2)$ sampled after burn-in')
    axes.scatter(weights[:, 1], weights[:, 2], marker='.', c="black")
    axes.plot(travel[:, 1], travel[:, 2], c="red")
    axes.set_xlabel(r'$w_2$')
    axes.set_ylabel(r'$w_1$')


def plot_bayesian_solution(axes, weights, data, labels):
    samples = np.linspace(2, 9, len(weights))
    axes.set_title('Bayesian solution')

    xmin, xmax = 1, 10
    ymin, ymax = 1, 8
    # For now the calculation only works if resolution is len(w). This should
    #  not be the case, so some calculation changes might be made.
    resolution = 100
    nsteps = complex(0, resolution)
    grid = np.mgrid[xmin:xmax:nsteps, ymin:ymax:nsteps]

    probability = orientation(weights, grid)

    axes.pcolormesh(grid[0], grid[1], probability)

    axes.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2],
                 marker='v', linewidths=3, c='black')
    axes.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2],
                 marker='o', linewidths=3, c='black')
    # axes.plot(samples,
    #           line(samples, weights[-10]),
    #           c="black")
    axes.set_ylim(1.9, 7.1)


def plotfig(samples, data, labels,
            show: bool = True, fname: str = ""):
    remove_first = 500

    m_values = np.zeros(len(samples[remove_first:]))
    for i, s in enumerate(samples[remove_first:]):
        m_values[i] = pstar.objective_function(s, data, labels, 0.01)

    travel = samples[:remove_first]
    weights = samples[remove_first:]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    plot_w_vs_iteration(ax[0, 0], weights)

    plot_m_vs_iteration(ax[0, 1], m_values)

    plot_spread(ax[1, 0], travel, weights)

    plot_bayesian_solution(ax[1, 1], weights, data, labels)

    if show:
        fig.show()
    else:
        if fname == "":
            fname = "mcmc_full.png"
        fig.savefig("{}{}".format(DATAFOLDER, fname))
