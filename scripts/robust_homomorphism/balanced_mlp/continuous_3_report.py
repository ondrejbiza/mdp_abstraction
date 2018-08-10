import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt


LOAD_DIR = "results/robust_homomorphism/balanced_mlp"
LOAD_FILE = "continuous_3_evaluation.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NUM_RUNS = 10
NUM_EXPERIENCE_LIST = [100, 200, 500, 1000, 2000, 5000]
SPLIT_THRESHOLD_LIST = [50, 100, 200, 500]

results = log_utils.read_pickle(LOAD_PATH)
results_array = np.zeros((len(NUM_EXPERIENCE_LIST), len(SPLIT_THRESHOLD_LIST)))

for i, num_experience in enumerate(NUM_EXPERIENCE_LIST):

    for j, split_threshold in enumerate(SPLIT_THRESHOLD_LIST):

        accuracies = []

        for run_idx in range(NUM_RUNS):

            key = (num_experience, split_threshold, run_idx)

            if key in results:

                accuracies.append(results[key])

        if len(accuracies) > 0:
            mean_accuracy = np.mean(accuracies)
            print(num_experience, split_threshold, mean_accuracy)
            results_array[i, j] = mean_accuracy

sns.heatmap(results_array, xticklabels=SPLIT_THRESHOLD_LIST, yticklabels=NUM_EXPERIENCE_LIST, annot=True, cbar=False)
plt.xlabel("split threshold")
plt.ylabel("num experience")
plt.show()