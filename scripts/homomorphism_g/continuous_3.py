import copy as cp
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from envs.continuous_3 import ContinuousEnv3
import continuous_homomorphism_g
import model_utils, vis_utils


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


def sample_actions(state):

    num = 10
    start = 0
    end = 2
    actions = list(np.linspace(start, end, num=num))
    return actions


def visualize_state_action_partition(state_action_partition):

    vis_utils.plot_background(env, show=False)
    vis_utils.plot_state_action_partition(state_action_partition, show=True)


def visualize_state_partition(state_partition):

    vis_utils.plot_background(env, show=False)
    vis_utils.plot_state_partition(state_partition, show=True)


env = ContinuousEnv3()

g = model_utils.GModel(DecisionTreeClassifier)


state_action_partition, state_partition = continuous_homomorphism_g.full_partition_iteration(
    lambda: gather_experience(env, 400), g, sample_actions, 1,
    visualize_state_action_partition=visualize_state_action_partition,
    visualize_state_partition=visualize_state_partition,
    max_iteration_steps=20
)