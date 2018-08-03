import numpy as np
import matplotlib.pyplot as plt


def plot_background(env, show=True):
    """
    Plot background for a state-action partition plot.
    :param env:         The environment.
    :param show:        Show the plot.
    :return:            None.
    """

    x = []
    for i in np.linspace(0, env.STATE_ACTION_MAP.shape[0] - 0.01, num=50):
        for j in np.linspace(0, env.STATE_ACTION_MAP.shape[1] - 0.01, num=50):
            x.append([i, j])

    x = np.array(x)
    y = np.array([env.get_state_action_block(p[0], p[1]) for p in x])

    plt.scatter(x[:, 0], x[:, 1], c=y, marker="s")

    if show:
        plt.show()


def plot_state_action_partition(state_action_partition, show=True):
    """
    Plot a state-action partition (assuming 1D actions and 1D states).
    :param state_action_partition:      State-action partition.
    :param show:                        Show the plot.
    :return:                            None.
    """

    for block in state_action_partition:
        x = np.empty((len(block), 2))
        for idx, t in enumerate(block):
            x[idx, 0] = t[0]
            x[idx, 1] = t[1]
        plt.scatter(x[:, 0], x[:, 1])

    if show:
        plt.show()
