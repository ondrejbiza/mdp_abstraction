import copy as cp
import numpy as np


def split(state_action_block, state_block, partition):

    partition = cp.deepcopy(partition)
    partition.remove(state_action_block)

    new_blocks = {}
    for state, action, reward, next_state, done in state_action_block:

        key = (reward, next_state in state_block)

        if key not in new_blocks.keys():
            new_blocks[key] = []

        new_blocks[key].append((state, action, reward, next_state, done))

    for new_block in new_blocks.values():
        partition.add(frozenset(new_block))

    return partition


def induce_state_partition(state_action_partition, distance, k):

    # get all states
    states = []
    blocks = []

    for idx, block in enumerate(state_action_partition):

        for state, action, reward, next_state, done in block:

            states.append(state)
            blocks.append(idx)
            states.append(next_state)
            blocks.append(idx)

    # create a distance matrix
    distances = np.empty((len(states), len(states)), dtype=np.float32)

    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):

            if j > i:
                break

            tmp_dist = distance(s1, s2)
            distances[i, j] = tmp_dist
            distances[j, i] = tmp_dist

    # k nearest neighbors classification
    state_partition = {}

    for idx, state in enumerate(states):

        tmp_blocks = set()
        nearest_neighbor_indexes = distances[idx].argsort()[:k]
        for nearest_neighbor_index in nearest_neighbor_indexes:
            block = blocks[nearest_neighbor_index]
            tmp_blocks.add(block)
        tmp_blocks = frozenset(tmp_blocks)
        if tmp_blocks not in state_partition:
            state_partition[tmp_blocks] = []
        state_partition[tmp_blocks].append(state)

    state_partition = {frozenset(value) for value in state_partition.values()}

    return state_partition


def partition_improvement(partition, distance, k, visualize_state_action_partition=None):

    new_partition = cp.deepcopy(partition)
    state_partition = induce_state_partition(new_partition, distance, k)

    for state_block in state_partition:

        flag = True

        while flag:

            flag = False

            for new_block in new_partition:

                tmp_new_partition = split(new_block, state_block, new_partition)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    if visualize_state_action_partition is not None:
                        print("split:")
                        visualize_state_action_partition(new_partition)

                    flag = True
                    break

    return new_partition


def partition_iteration(partition, distance, k, max_steps=2, visualize_state_action_partition=None):

    new_partition = partition_improvement(
        partition, distance, k, visualize_state_action_partition=visualize_state_action_partition
    )
    step = 1

    while partition != new_partition and step < max_steps:

        partition = new_partition
        new_partition = partition_improvement(
            partition, distance, k, visualize_state_action_partition=visualize_state_action_partition
        )

        step += 1

    return new_partition


def get_experience(state_action_partition):

    experience = []

    for block in state_action_partition:
        for t in block:
            experience.append(t)

    return experience


def full_partition_iteration(gather_experience, distance, k, num_steps,
                             visualize_state_action_partition=None, visualize_state_partition=None):

    state_action_partition = set()

    for step in range(num_steps):

        # collect experience
        experience = gather_experience()

        # add experience
        experience += get_experience(state_action_partition)
        state_action_partition = {frozenset(experience)}

        # visualize added experience
        if visualize_state_action_partition is not None:
            visualize_state_action_partition(state_action_partition)

        # rearrange partition
        state_action_partition = partition_iteration(state_action_partition, distance, k,
                                                     visualize_state_action_partition=visualize_state_action_partition)

        # visualize rearranged partition
        if visualize_state_action_partition is not None:
            print("step {:d} state-action partition:".format(step + 1))
            visualize_state_action_partition(state_action_partition)

        # visualize state partition
        if visualize_state_partition is not None:
            print("step {:d} state partition:".format(step + 1))
            state_partition = induce_state_partition(state_action_partition, distance, k)
            visualize_state_partition(state_partition)

    state_partition = induce_state_partition(state_action_partition, distance, k)

    return state_action_partition, state_partition