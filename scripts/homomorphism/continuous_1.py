import copy as cp
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from envs.continuous_1 import ContinuousEnv1
import continuous_homomorphism


class CustomLogisticRegression:

    def __init__(self):

        self.logistic_regression = LogisticRegression()
        self.single_class = True

    def predict(self, state):

        if self.single_class:
            return 0
        else:
            return self.logistic_regression.predict(state)[0]

    def fit(self, x, y):

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        if np.all(y == 0):
            self.single_class = True
        else:
            self.single_class = False
            self.logistic_regression.fit(x, y)


def gather_experience(env, num):

    transitions = []

    for _ in range(num // 5):

        env.reset()

        for _ in range(5):

            state = cp.deepcopy(env.state)
            action = np.random.uniform(0, 2)
            reward, next_state, done = env.step(action)

            transitions.append((state, action, reward, next_state, done))

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

d = env.state_distance
k = 10
f = CustomLogisticRegression()


state_action_partition, state_partition = continuous_homomorphism.full_partition_iteration(lambda: gather_experience(env, 100), f, d, k, 10, reclassify=True)

print("num state action partitions:", len(state_action_partition))
print("num state partitions:", len(state_partition))

for block in state_action_partition:
    x = np.empty((len(block), 2))
    for idx, t in enumerate(block):
        x[idx, 0] = t[0]
        x[idx, 1] = t[1]
    plt.scatter(x[:, 0], x[:, 1])
plt.show()