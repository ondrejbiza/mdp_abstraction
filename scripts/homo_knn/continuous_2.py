import copy as cp
import numpy as np
from envs.continuous_2 import ContinuousEnv2
from algorithms import online_homomorphism_knn
import vis_utils


def gather_experience(env, num):

    transitions = []

    for _ in range(num // 5):

        env.reset()

        for _ in range(5):

            state = cp.deepcopy(env.state)
            action = np.random.uniform(0, 2)
            reward, next_state, done = env.step(action)

            transitions.append((state, action, reward, next_state, done))

            if done:
                break

    return transitions


def visualize_state_action_partition(state_action_partition):

    vis_utils.plot_background(env, show=False)
    vis_utils.plot_state_action_partition(state_action_partition, show=True)


def visualize_state_partition(state_partition):

    vis_utils.plot_background(env, show=False)
    vis_utils.plot_state_partition(state_partition, show=True)


env = ContinuousEnv2()

d = env.state_distance
k = 10


state_action_partition, state_partition = online_homomorphism_knn.full_partition_iteration(
    lambda: gather_experience(env, 400), d, k, 1,
    visualize_state_action_partition=visualize_state_action_partition,
    visualize_state_partition=visualize_state_partition
)