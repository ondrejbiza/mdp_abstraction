import argparse
import copy as cp
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from algorithms.online_homomorphism_g_dict import OnlineHomomorphismGDict
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

    return transitions


def sample_actions(state):

    num = 10
    start = 0
    end = 2
    actions = list(np.linspace(start, end, num=num))
    return actions


def main(args):

    if args.environment == 1:
        from envs.continuous_1 import ContinuousEnv1 as Env
    elif args.environment == 2:
        from envs.continuous_2 import ContinuousEnv2 as Env
    else:
        from envs.continuous_3 import ContinuousEnv3 as Env

    env = Env()

    def visualize_b(state_action_partition):

        vis_utils.plot_background(env, show=False)
        vis_utils.plot_state_action_partition(state_action_partition, show=True)

    g = model_utils.GModel(DecisionTreeClassifier())

    experience = gather_experience(env, args.num_experience)

    homo = OnlineHomomorphismGDict(experience, g, sample_actions, args.b_threshold,
                                   OnlineHomomorphismGDict.RESOLVE_ADD_CLOSEST, 20, visualize_b=visualize_b)
    homo.partition_iteration()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("environment", type=int, help="1, 2 or 3; increasing in difficulty")

    parser.add_argument("--num-experience", type=int, default=400)
    parser.add_argument("--b-threshold", type=int, default=50)

    parsed = parser.parse_args()
    main(parsed)