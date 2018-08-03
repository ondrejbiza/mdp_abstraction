import copy as cp
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from envs.continuous_1 import ContinuousEnv1
import continuous_homomorphism
import visualize


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


env = ContinuousEnv1()

d = env.state_distance
k = 10
f = CustomLogisticRegression()


state_action_partition, state_partition = continuous_homomorphism.full_partition_iteration(lambda: gather_experience(env, 100), f, d, k, 2, reclassify=False)

print("num state action partitions:", len(state_action_partition))
print("num state partitions:", len(state_partition))

visualize.plot_background(env, show=False)
visualize.plot_state_action_partition(state_action_partition, show=True)