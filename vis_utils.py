import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


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


def show_confidences(state_action_partition, classifier, sample_actions, show=True):
    """
    Show confidences together with real data points as an linearly interpolated image.
    :param state_action_partition:  State-action partition.
    :param classifier:              State-action classifier.
    :param sample_actions:          Function that samples actions given a state.
    :param show:                    Show plot.
    :return:                        None.
    """

    xs = []
    ys = []
    colors = []

    real_points = []
    real_colors = []

    for idx, block in enumerate(state_action_partition):
        for transition in block:

            real_points.append([transition[0], transition[1]])
            real_colors.append(idx)

            state = transition[3]
            actions = sample_actions(state)

            probs = classifier.batch_predict_prob([state] * len(actions), actions)

            for i in range(len(probs)):
                xs.append(state)
                ys.append(actions[i])
                colors.append(probs[i][np.argmax(probs[i])])

    real_points = np.array(real_points, dtype=np.float32)
    real_colors = np.array(real_colors, dtype=np.float32)

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    x_min = int(np.floor(np.min(xs)))
    x_max = int(np.ceil(np.max(xs)))
    y_min = int(np.floor(np.min(ys)))
    y_max = int(np.ceil(np.max(ys)))

    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    grid = griddata(np.stack([xs, ys], axis=-1), colors, (grid_x, grid_y), method="linear")

    plt.figure(figsize=(14, 8))

    plt.imshow(grid.T, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="gray", vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label("confidence")

    plt.scatter(real_points[:, 0], real_points[:, 1], c=real_colors)

    plt.xlabel("states")
    plt.ylabel("actions")

    if show:
        plt.show()
