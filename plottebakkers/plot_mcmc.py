import matplotlib.pyplot as plt

DATAFOLDER = "../data/"


def line(x, w0, w1, w2):
    return -(w1 * x + w0) / w2


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


def plot_bayesian_solution(axes, weights, data, samples, labels):
    axes.set_title('Bayesian solution')
    axes.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2],
                 marker='+', linewidths=5, c='b')
    axes.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2],
                 marker='o', linewidths=3, c='r')
    axes.plot(samples,
              line(samples, weights[-10, 0], weights[-10, 1], weights[-10, 2]),
              marker=',')
    axes.set_ylim(1.9, 7.1)


def plotfig(weights, m_values, data, samples, labels,
            show: bool = True, fname: str = ""):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    plot_w_vs_iteration(ax[0, 0], weights)

    plot_m_vs_iteration(ax[0, 1], m_values)

    plot_spread(ax[1, 0], weights)

    plot_bayesian_solution(ax[1, 1], weights, data, samples, labels)

    if show:
        fig.show()
    else:
        if fname == "":
            fname = "mcmc_full.png"
        fig.savefig("{}{}".format(DATAFOLDER, fname))
