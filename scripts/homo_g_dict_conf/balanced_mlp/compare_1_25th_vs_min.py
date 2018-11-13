import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt

LOAD_DIR_1 = "results/homo_g_dict_conf/balanced_mlp"
LOAD_FILE_1 = "experiment_1_thresholds_25th_percentile.pickle"
LOAD_PATH_1 = os.path.join(LOAD_DIR_1, LOAD_FILE_1)

LOAD_DIR_2 = "results/homo_g_dict_conf/balanced_mlp"
LOAD_FILE_2 = "experiment_1_thresholds.pickle"
LOAD_PATH_2 = os.path.join(LOAD_DIR_2, LOAD_FILE_2)

NUM_RUNS = 200
NUM_EXPERIENCE_LIST = [200, 500, 1000]
SPLIT_THRESHOLD_LIST = [10, 20, 50]
CONF_THRESHOLD_LIST = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]

results_1 = log_utils.read_pickle(LOAD_PATH_1)
results_2 = log_utils.read_pickle(LOAD_PATH_2)

results_array = np.zeros((len(NUM_EXPERIENCE_LIST), len(SPLIT_THRESHOLD_LIST), len(CONF_THRESHOLD_LIST)))

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    for j, split_threshold in enumerate(SPLIT_THRESHOLD_LIST):

        for k, min_conf in enumerate(CONF_THRESHOLD_LIST):

            accuracies_1 = []
            accuracies_2 = []

            for run_idx in range(NUM_RUNS):

                key = (num_experience, split_threshold, min_conf, run_idx)

                if key in results_1:
                    accuracies_1.append(results_1[key])

                if key in results_2:
                    accuracies_2.append(results_2[key])

            if len(accuracies_1) > 0 and len(accuracies_2):

                mean_accuracy_1 = np.mean(accuracies_1)
                mean_accuracy_2 = np.mean(accuracies_2)
                ratio = np.round((mean_accuracy_1 / mean_accuracy_2) - 1, decimals=3)

                print(num_experience, split_threshold, min_conf, ratio)
                results_array[i, j, k] = ratio

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    sns.heatmap(results_array[i, :, :], xticklabels=CONF_THRESHOLD_LIST, yticklabels=SPLIT_THRESHOLD_LIST, annot=True,
                cbar=False)
    plt.title("{:d} transitions".format(num_experience))
    plt.xlabel("minimum confidence threshold for state blocks")
    plt.ylabel("split threshold for state-action blocks")
    plt.show()
