import argparse
import copy as cp
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from envs.continuous_3 import ContinuousEnv3
from algorithms import online_homomorphism_g
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


def main(args):

    env = ContinuousEnv3()

    def visualize_b(state_action_partition):
        vis_utils.plot_background(env, show=False)
        vis_utils.plot_state_action_partition(state_action_partition, show=True)

    def visualize_sb(state_partition):
        vis_utils.plot_background(env, show=False)
        vis_utils.plot_state_partition(state_partition, show=True)

    g = model_utils.GModel(DecisionTreeClassifier())

    experience = gather_experience(env, args.num_experience)
    homo = online_homomorphism_g.OnlineHomomorphismG(experience, g, sample_actions, 1, 1, 20, visualize_b=visualize_b,
                                                     visualize_sb=visualize_sb)
    homo.partition_iteration()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-experience", type=int, default=400)

    parsed = parser.parse_args()
    main(parsed)