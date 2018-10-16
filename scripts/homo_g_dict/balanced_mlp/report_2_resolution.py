import os
import numpy as np
import log_utils
import seaborn as sns
import matplotlib.pyplot as plt
from algorithms.online_homomorphism_g_dict import OnlineHomomorphismGDict

LOAD_DIR = "results/homo_g_dict/balanced_mlp"
LOAD_FILE = "experiment_2_resolution.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NUM_RUNS = 50
RESOLUTION_LIST = [OnlineHomomorphismGDict.RESOLVE_IGNORE, OnlineHomomorphismGDict.RESOLVE_ADD_CLOSEST,
                   OnlineHomomorphismGDict.RESOLVE_ADD_TO_RANDOM, OnlineHomomorphismGDict.RESOLVE_ADD_TO_BIGGEST]

results = log_utils.read_pickle(LOAD_PATH)
results_array = np.zeros(len(RESOLUTION_LIST))

for i, resolution in enumerate(RESOLUTION_LIST):

    accuracies = []

    for run_idx in range(NUM_RUNS):

        key = (resolution, run_idx)

        if key in results:

            accuracies.append(results[key])

    if len(accuracies) > 0:
        mean_accuracy = np.mean(accuracies)
        print(resolution, mean_accuracy)
        results_array[i] = mean_accuracy

sns.barplot(results_array)
plt.xlabel("split threshold for state-action blocks")
plt.ylabel("num experience")
plt.show()
