import argparse
import os
import copy as cp
import numpy as np
from envs.continuous_3 import ContinuousEnv3
from algorithms.online_homomorphism_g_conf import OnlineHomomorphismG
import evaluation, model_utils, log_utils

NUM_RUNS = 200
NUM_EXPERIENCE_LIST = [200, 500, 1000]
SPLIT_THRESHOLD_LIST = [10, 20, 50]
CONF_THRESHOLD_LIST = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]
PERCENTILE = 12
SAVE_DIR = "results/homo_g_conf/balanced_mlp"
SAVE_FILE = "experiment_1_thresholds_12th_percentile.pickle"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE)


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


def run(num_experience, split_threshold, min_confidence):

    env = ContinuousEnv3()
    g = model_utils.BalancedMLP([1], [8, 16], 0.001, 32, 0.0, verbose=True)

    experience = gather_experience(env, num_experience)
    homo = OnlineHomomorphismG(experience, g, sample_actions, split_threshold, 0, 20, min_confidence,
                               percentile=PERCENTILE)
    homo.partition_iteration()

    hits, total = evaluation.overlap(env, list(homo.partition))
    accuracy = hits / total

    return accuracy


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

            for min_conf in CONF_THRESHOLD_LIST:

                for run_idx in range(NUM_RUNS):

                    key = (num_experience, split_threshold, min_conf, run_idx)

                    # skip if this setting is already in results
                    if key in results:
                        continue

                    accuracy = run(num_experience, split_threshold, min_conf)
                    results[key] = accuracy

                    # save results after each run
                    log_utils.write_pickle(SAVE_PATH, results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed = parser.parse_args()
    main(parsed)
