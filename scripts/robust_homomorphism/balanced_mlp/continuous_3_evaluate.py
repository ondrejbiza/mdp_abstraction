import argparse
import os
import copy as cp
import numpy as np
from envs.continuous_3 import ContinuousEnv3
import robust_homomorphism
import evaluation, model_utils, log_utils


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


def run(num_experience, split_threshold):

    env = ContinuousEnv3()
    g = model_utils.BalancedMLP([1], [8, 16], 0.001, 32, 0.0, verbose=True)

    state_action_partition, state_partition = robust_homomorphism.full_partition_iteration(
        lambda: gather_experience(env, num_experience), g, sample_actions, 1, split_threshold,
        max_iteration_steps=20
    )

    hits, total = evaluation.overlap(env, list(state_action_partition))
    accuracy = hits / total

    return accuracy


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    NUM_RUNS = 10
    NUM_EXPERIENCE_LIST = [100, 200, 500, 1000, 2000, 5000]
    SPLIT_THRESHOLD_LIST = [50, 100, 200, 500]
    SAVE_DIR = "results/robust_homomorphism/balanced_mlp"
    SAVE_FILE = "continuous_3_evaluation.pickle"
    SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE)

    # maybe create dir
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # load results if already exist
    if os.path.isfile(SAVE_PATH):
        results = log_utils.read_pickle(SAVE_PATH)
    else:
        results = {}

    for num_experience in NUM_EXPERIENCE_LIST:

        for split_threshold in SPLIT_THRESHOLD_LIST:

            for run_idx in range(NUM_RUNS):

                key = (num_experience, split_threshold, run_idx)

                # skip if this setting is already in results
                if key in results:
                    continue

                accuracy = run(num_experience, split_threshold)
                results[key]= accuracy

                # save results after each run
                log_utils.write_pickle(SAVE_PATH, results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed = parser.parse_args()
    main(parsed)