import matplotlib.pyplot as plt
import numpy as np
import week_2.p_star_distribution as pstar

DATAFOLDER = "./data/"


def line(grid, weights):
    # generate the xgrid (tranposed by default)
    xgrid_transposed = grid[0, :, :]
    return -(weights[:, 1] * xgrid_transposed + weights[:, 0]) / weights[:, 2]


def plot_w_vs_iteration(axes, weights):
    axes.set_title(r'$w$ vs iteration')
    axes.plot(weights[:, 0], label=r'$w_1$')
    axes.plot(weights[:, 1], label=r'$w_2$')
    axes.plot(weights[:, 2], label=r'$w_3$')
    axes.legend()


def plot_m_vs_iteration(axes, m_values):
    axes.set_title(r'$M$ vs iteration')
    axes.plot(m_values)


def plot_spread(axes, weights, travel=None,
                title: str = ""):
    if title == "":
        axes.set_title(r'$(w_1, w_2)$ sampled after burn-in')
    else:
        axes.set_title(title)
    axes.plot(weights[:, 1], weights[:, 2], marker=',', c="black", linestyle='')
    if travel is not None:
        axes.plot(travel[:, 1], travel[:, 2], c="red")
    axes.set_xlabel(r'$w_2$')
    axes.set_ylabel(r'$w_1$')


def plot_bayesian_solution(axes, data, labels, grid, probability):
    axes.set_title('Bayesian solution')

    axes.pcolormesh(grid[0], grid[1], probability, cmap='RdBu')

    axes.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2],
                 marker='v', linewidths=7, c='white')
    axes.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2],
                 marker='o', linewidths=7, c='white')
    axes.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2],
                 marker='v', linewidths=3, c='black')
    axes.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2],
                 marker='o', linewidths=3, c='black')
    axes.set_ylim(1.9, 7.1)


def determine_travel(accept_values):
    """
    Determines the length of the travel distance by taking the std of the accept
    values and determining the derivative of this.
    """
    # Calculate the progressing std of the accept values, in the beginning this
    #  should be chaotic, but eventually the std should decrease since the
    #  accept values will reach values between 0 and 1.
    stds = np.zeros(len(accept_values))
    for i in range(len(accept_values)):
        stds[i] = accept_values[:i].std()

    # Calculate the derivative of these std values. We want to find the location
    #  where the std starts to descent fluidly, this is where the accept values
    #  start to reach the values between 0 and 1 (ignore the first one since
    #  this is always nan).
    std_diff = np.diff(stds)[1:]

    # The descent starts once the std has a negative derivative. However, it can
    #  happen that there will be a spike in the accept value, so locate the
    #  spike, and repeat the process of finding the minimum.
    start_descent = np.argmax(stds) + 2
    while np.max(std_diff[start_descent:]) > 1:
        start_descent += np.argmax(std_diff[start_descent:])
        start_descent += np.argmin(std_diff[start_descent:]) + 1

    # Skip to the place where the accept values will not deviate that strongly
    #  anymore.
    length = np.argmax(std_diff[start_descent:] > -3.5) + start_descent + 1

    return length, stds


def plot_travel(samples, accept_values,
                show: bool = True, file_name: str = ''):
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
    travel_length, history = determine_travel(accept_values)

    travel = samples[:travel_length]
    weights = samples[travel_length:]

    ax[0].set_title(r"$\sigma(a)$ up to $x$ vs. $x$")
    ax[0].set_yscale('log')
    ax[0].set_xlabel('number of samples')
    ax[0].set_ylabel(r"$\sigma(a)$")

    x_pre_travel = np.arange(travel_length)
    x_post_travel = np.arange(travel_length, len(samples))
    ax[0].plot(x_pre_travel, history[:travel_length], c="red")
    ax[0].plot(x_post_travel, history[travel_length:], c="black")
    plot_spread(ax[1], weights, travel)

    fig.tight_layout()

    if show:
        fig.show()
    else:
        if file_name == '':
            file_name = 'log_std_vs_travel.png'
        destination = '{}{}'.format(DATAFOLDER, file_name)
        fig.tight_layout()
        fig.savefig(destination)


def plot_prob_difference(grid, prob_grid_1, prob_grid_2,
                         show: bool = True, filename: str = ""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Difference in probability maps.")
    diff = prob_grid_1 - prob_grid_2
    mappable = ax.pcolormesh(grid[0], grid[1], diff, cmap='seismic')
    fig.colorbar(mappable, ax=ax)
    if show:
        fig.show()
    else:
        if filename == "":
            filename = "diff_mh_ham.png"
        fig.tight_layout()
        fig.savefig("{}{}".format(DATAFOLDER, filename))


def plotfig(samples, data, labels, probability_grid, grid, accept_values=None,
            show: bool = True, fname: str = ""):
    if accept_values is not None:
        travel_length, dump = determine_travel(accept_values)
    else:
        travel_length = 0

    m_values = np.zeros(len(samples[travel_length:]))
    for i, s in enumerate(samples[travel_length:]):
        m_values[i] = pstar.objective_function(s, data, labels, 0.01)

    travel = samples[:travel_length]
    weights = samples[travel_length:]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    plot_w_vs_iteration(ax[0, 0], weights)

    plot_m_vs_iteration(ax[0, 1], m_values)

    plot_spread(ax[1, 0], weights, travel)

    plot_bayesian_solution(ax[1, 1], data, labels, grid, probability_grid)

    if show:
        fig.show()
    else:
        if fname == "":
            fname = "mcmc_full.png"
        fig.tight_layout()
        fig.savefig("{}{}".format(DATAFOLDER, fname))
