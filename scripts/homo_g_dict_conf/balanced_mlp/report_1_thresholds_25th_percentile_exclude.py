import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt

LOAD_DIR = "results/homo_g_dict_conf/balanced_mlp"
LOAD_FILE = "experiment_1_thresholds_25th_percentile_exclude.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NUM_RUNS = 200
NUM_EXPERIENCE_LIST = [200, 500, 1000]
SPLIT_THRESHOLD_LIST = [10, 20, 50]
CONF_THRESHOLD_LIST = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]

results = log_utils.read_pickle(LOAD_PATH)
results_array = np.zeros((len(NUM_EXPERIENCE_LIST), len(SPLIT_THRESHOLD_LIST), len(CONF_THRESHOLD_LIST)))

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    for j, split_threshold in enumerate(SPLIT_THRESHOLD_LIST):

        for k, min_conf in enumerate(CONF_THRESHOLD_LIST):

            accuracies = []

            for run_idx in range(NUM_RUNS):

                key = (num_experience, split_threshold, min_conf, run_idx)

                if key in results:

                    accuracies.append(results[key])

            if len(accuracies) > 0:
                mean_accuracy = np.mean(accuracies)
                print(num_experience, split_threshold, min_conf, mean_accuracy)
                results_array[i, j, k] = mean_accuracy

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    sns.heatmap(results_array[i, :, :], xticklabels=CONF_THRESHOLD_LIST, yticklabels=SPLIT_THRESHOLD_LIST, annot=True,
                cbar=False)
    plt.title("{:d} transitions".format(num_experience))
    plt.xlabel("minimum confidence threshold for state blocks")
    plt.ylabel("split threshold for state-action blocks")
    plt.show()
