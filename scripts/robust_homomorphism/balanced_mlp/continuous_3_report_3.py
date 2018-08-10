import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt


LOAD_DIR = "results/robust_homomorphism/balanced_mlp"
LOAD_FILE = "continuous_3_evaluation_3.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NUM_RUNS = 10
SPLIT_THRESHOLD_LIST = [50, 100, 200, 500]

results = log_utils.read_pickle(LOAD_PATH)
results_array = np.zeros((len(SPLIT_THRESHOLD_LIST), len(SPLIT_THRESHOLD_LIST)))

for i, t1 in enumerate(SPLIT_THRESHOLD_LIST):

    for j, t2 in enumerate(SPLIT_THRESHOLD_LIST):

        accuracies = []

        for run_idx in range(NUM_RUNS):

            key = (t1, t2, run_idx)

            if key in results:

                accuracies.append(results[key])

        if len(accuracies) > 0:
            mean_accuracy = np.mean(accuracies)
            print(t1, t2, mean_accuracy)
            results_array[i, j] = mean_accuracy

sns.heatmap(results_array, xticklabels=SPLIT_THRESHOLD_LIST, yticklabels=SPLIT_THRESHOLD_LIST, annot=True, cbar=False)
plt.ylabel("split threshold for state-action blocks")
plt.xlabel("split threshold for state blocks")
plt.show()