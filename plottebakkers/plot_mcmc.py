import matplotlib.pyplot as plt
import numpy as np

DATAFOLDER = "../data/"


def line(grid, weights):
    # generate the xgrid (tranposed by default)
    xgrid_transposed = grid[0, :, :]
    return -(weights[:, 1] * xgrid_transposed + weights[:, 0]) / weights[:, 2]


def outer_points(grid, weights):
    """
    Calculate the rightmost and leftmost points of the lines created by the
    weights delivered.
    :param grid: grid used to generate the lines in.
    :param weights: the weights used to calculate the lines, where the lines
    go by the function -(w[1] * x + w[0]) / w[2].
    :return: Array of shape (len(grid), 4) with the first to indicating the
    point p with the minimal x- and y-value, and the last two indicating point
    q with max x- and y-value.
    """
    x_lims = np.array([grid[0, 0, :], grid[0, -1, :]])
    line_lims = -(weights[:, 1] * x_lims + weights[:, 0]) / weights[:, 2]

    mins = np.array([x_lims[0], line_lims[0]]).transpose()
    maxs = np.array([x_lims[1], line_lims[1]]).transpose()

    return np.concatenate((mins, maxs), axis=1)


def orientation(p, q, r):
    """
    Calculates the orientation of point r with the line between point p and q
    :param p: 2D (x, y) leftmost point of the line
    :param q: 2D (x, y) rightmost point of the line
    :param r: 2D (x, y) point where the orientation must be determined of
    :return: NxN grid with percentages of how much lines are right to the points
    """
    # len-4000 vectors
    dx = q[0] - p[0]
    dy = q[1] - p[1]

    # size 4000, 4000 matrix (the grid)
    r_xdiff = (r[0] - q[0][0])
    r_ydiff = (r[1] - q[1][0])
    cnt = np.zeros(shape=(len(r_xdiff), len(r_ydiff)))

    print("|", end='')
    for i, (x, y) in enumerate(zip(dx, dy)):
        if i / len(q[0]) * 100 % 5 == 0:
            print("█", end='')
        val = y * r_xdiff - x * r_ydiff
        cnt += np.sign(val)

    print("|")

    # Scale the array between 0 and 1, 0 meaning all right.
    percentage_clockwise = cnt / (2 * np.max(cnt)) + 0.5

    return percentage_clockwise


def propability_grid(weights):
    x_min, x_max = 1, 10
    y_min, y_max = 1, 8
    n_steps = complex(0, len(weights))
    grid = np.mgrid[x_min:x_max:n_steps, y_min:y_max:n_steps]

    x_range = grid[0, :, 0]
    y_range = grid[1, 0, :]

    # generate array of lines, where the first index indicates the line, and the
    #  second index indicates the y-coordinate of the line.
    lines = line(grid, weights)
    op = outer_points(grid, weights)


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

    xmin, xmax = 1, 10
    ymin, ymax = 1, 8
    # For now the calculation only works if resolution is len(w). This should
    #  not be the case, so some calculation changes might be made.
    resolution = len(weights)
    nsteps = complex(0, resolution)
    grid = np.mgrid[xmin:xmax:nsteps, ymin:ymax:nsteps]

    # Generate array containing the rightmost and leftmost points of all the
    #  lines.
    ops = outer_points(grid, weights)
    # Split the array into two: the rightmost (p) and the leftmost (q)
    p = ops[:, :2].transpose()
    q = ops[:, 2:].transpose()

    # Calculate the probability grid of how many lines are clockwise to each
    #  coordinate in the grid.
    probability = orientation(p, q, grid)

    axes.pcolormesh(grid[0], grid[1], probability)

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
