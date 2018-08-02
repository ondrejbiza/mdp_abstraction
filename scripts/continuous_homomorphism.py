import numpy as np
import sklearn
import matplotlib.pyplot as plt
import envs.continuous as env


def gather_experience(num):

    transitions = []

    for _ in range(num):

        action = np.random.uniform(0, env.map.shape[0])
        state = np.random.uniform(0, env.map.shape[1])
        reward, next_state = env.step(state, action)

        transitions.append((state, action, reward, next_state))

    return transitions


def plot_experience(experience):

    points = [[state, action] for state, action, _, _ in experience]
    points = np.array(points, dtype=np.float32)

    x = []
    for i in np.linspace(0, 5.99, num=100):
        for j in np.linspace(0, 6.99, num=100):
            x.append([i, j])

    x = np.array(x)
    y = np.array([env.get_block(p[0], p[1]) for p in x])

    plt.scatter(x[:, 0], x[:, 1], c=y, marker="s")
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


