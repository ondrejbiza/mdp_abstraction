import argparse
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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
    g = model_utils.GModel(MLPClassifier(hidden_layer_sizes=(32, 32, 32), solver="sgd", batch_size=128,
                                         learning_rate_init=0.1, learning_rate="adaptive", max_iter=2000,
                                         early_stopping=False))

    def visualize_b(state_action_partition):
        vis_utils.plot_background(env, show=False)

        xx, yy = np.meshgrid(np.arange(0, env.STATE_ACTION_MAP.shape[1], 0.01),
                             np.arange(0, env.STATE_ACTION_MAP.shape[0], 0.01))
        data = np.c_[xx.ravel(), yy.ravel()]
        Z = g.batch_predict(data[:, 0], data[:, 1])
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)

        vis_utils.plot_state_action_partition(state_action_partition, show=True)

    def visualize_sb(state_partition):
        vis_utils.plot_background(env, show=False)
        vis_utils.plot_state_partition(state_partition, show=True)

    experience = gather_experience(env, args.num_experience)
    homo = online_homomorphism_g.OnlineHomomorphismG(experience, g, sample_actions, 1, 1, 20, visualize_b=visualize_b,
                                                     visualize_sb=visualize_sb)
    homo.partition_iteration()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-experience", type=int, default=2000)

    parsed = parser.parse_args()
    main(parsed)