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
    for i in np.linspace(0, env.STATE_ACTION_MAP.shape[1] - 0.01, num=50):
        for j in np.linspace(0, env.STATE_ACTION_MAP.shape[0] - 0.01, num=50):
            x.append([i, j])

    x = np.array(x)
    y = np.array([env.get_state_action_block(p[0], p[1]) for p in x])

    plt.scatter(x[:, 0], x[:, 1], c=y, marker="s", cmap="Greys", vmin=-2, vmax=8)

    if show:
        plt.show()


def plot_state_action_partition(state_action_partition, show=True):
    """
    Plot a state-action partition (assuming 1D actions and 1D states).
    :param state_action_partition:      State-action partition.
    :param show:                        Show the plot.
    :return:                            None.
    """

    x = []
    c = []

    for idx, block in enumerate(state_action_partition):
        for transition in block:
            x.append([transition[0], transition[1]])
            c.append(idx)

    x = np.array(x, dtype=np.float32)
    c = np.array(c, dtype=np.int32)

    plt.scatter(x[:, 0], x[:, 1], c=c, cmap="hot", vmin=0, vmax=np.max(c) + 2)
    plt.xlabel("States")
    plt.ylabel("Actions")

    if show:
        plt.show()


def plot_state_partition(state_partition, show=True):
    """
    Plot state partition.
    :param state_partition:     State partition.
    :param show:                Show plot.
    :return:                    None.
    """

    for block in state_partition:
        x = np.ones((len(block), 2))
        for idx, state in enumerate(block):
            x[idx, 0] = state
        plt.scatter(x[:, 0], x[:, 1])

    if show:
        plt.show()


def plot_decision_boundary(classifier, height, width, show=True):
    """
    Plot decision boundary for a classifier.
    :param classifier:      A classifier.
    :param height:          Height of the 2D input space.
    :param width:           Width of the 2D input space.
    :param show:            Show plot.
    :return:                None.
    """

    xx, yy = np.meshgrid(np.arange(0, height, 0.01), np.arange(0, width, 0.01))
    data = np.c_[xx.ravel(), yy.ravel()]
    z = classifier.batch_predict(data[:, 0], data[:, 1])
    z = np.array(z).reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.4)

    if show:
        plt.show()
