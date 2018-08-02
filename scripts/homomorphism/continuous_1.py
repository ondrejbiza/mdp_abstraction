import numpy as np
import sklearn
import matplotlib.pyplot as plt
from envs.continuous_1 import ContinuousEnv1


def gather_experience(env, num):

    transitions = []

    for _ in range(num // 5):

        env.reset()

        for _ in range(5):

            state = np.copy(env.state)
            action = np.random.uniform(0, 2)
            reward, next_state = env.step(action)

            transitions.append((state, action, reward, next_state))

    return transitions


def plot_experience(env, experience):

    points = [[state, action] for state, action, _, _ in experience]
    points = np.array(points, dtype=np.float32)

    x = []
    for i in np.linspace(0, 1.99, num=100):
        for j in np.linspace(0, 1.99, num=100):
            x.append([i, j])

    x = np.array(x)
    y = np.array([env.get_state_action_block(p[0], p[1]) for p in x])

    plt.scatter(x[:, 0], x[:, 1], c=y, marker="s")
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


env = ContinuousEnv1()
exp = gather_experience(env, 100)
plot_experience(env, exp)