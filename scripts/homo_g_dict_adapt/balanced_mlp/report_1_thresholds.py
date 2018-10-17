import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt

LOAD_DIR = "results/homo_g_dict_adapt/balanced_mlp"
LOAD_FILE = "experiment_1_thresholds.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NUM_RUNS = 50
NUM_EXPERIENCE_LIST = [200, 500, 1000]
THRESHOLD_MULTIPLIER_LIST = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

results = log_utils.read_pickle(LOAD_PATH)
results_array = np.zeros((len(NUM_EXPERIENCE_LIST), len(THRESHOLD_MULTIPLIER_LIST)))

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    for j, split_threshold in enumerate(THRESHOLD_MULTIPLIER_LIST):

        accuracies = []

        for run_idx in range(NUM_RUNS):

            key = (num_experience, split_threshold, run_idx)

            if key in results:

                accuracies.append(results[key])

        if len(accuracies) > 0:
            mean_accuracy = np.mean(accuracies)
            print(num_experience, split_threshold, mean_accuracy)
            results_array[i, j] = mean_accuracy

sns.heatmap(results_array, xticklabels=THRESHOLD_MULTIPLIER_LIST, yticklabels=NUM_EXPERIENCE_LIST, annot=True, cbar=False)
plt.xlabel("split threshold for state-action blocks")
plt.ylabel("num experience")
plt.show()
