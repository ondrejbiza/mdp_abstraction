import numpy as np


def overlap(env, state_action_partition):

    num_gt_partitions = np.max(env.MINIMAL_STATE_ACTION_MAP) + 1
    num_predicted_partition = len(state_action_partition)

    # count experience
    total = 0
    for block in state_action_partition:
        total += len(block)

    # calculate number of matches for each ground-truth block, predicted block pair
    num_matches = np.zeros((num_gt_partitions, num_predicted_partition), dtype=np.int32)
    for i in range(num_gt_partitions):

        for j in range(num_predicted_partition):

            for transition in state_action_partition[j]:

                actual_block = env.get_state_action_block_minimal_map(transition[0], transition[1])

                if actual_block == i:

                    num_matches[i, j] += 1

    # match ground-truth partitions to predicted partitions
    matches = {}
    hits = 0

    for _ in range(min(num_gt_partitions, num_predicted_partition)):

        coords = np.unravel_index(num_matches.argmax(), num_matches.shape)
        matches[coords[0]] = coords[1]

        hits += num_matches[coords]

        num_matches[coords[0], :] = -1
        num_matches[:, coords[1]] = -1

    return hits, total