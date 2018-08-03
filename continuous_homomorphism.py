import random
import copy as cp
import numpy as np


def split(state_action_block, state_block, partition, classifier):

    partition = cp.deepcopy(partition)
    partition.remove(state_action_block)

    new_blocks = {}
    for state, action, reward, next_state, done in state_action_block:

        next_state_block = classifier.predict(next_state)
        key = (reward, next_state_block == state_block)

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

            if done:
                # add the final state to the state partition
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


def get_state_partition(state_action_partition, classifier):

    # get all states
    states = set()

    for state_action_block in state_action_partition:

        for state, action, reward, next_state, done in state_action_block:

            states.add(state)

            if done:
                states.add(next_state)

    # sort states into partitions
    state_partition = {}

    for state in states:

        state_block = classifier.predict(state)

        if state_block not in state_partition:
            state_partition[state_block] = []

        state_partition[state_block].append(state)

    state_partition = {frozenset(value) for value in state_partition.values()}

    return state_partition


def train_f(state_partition, f):

    x = []
    y = []

    for idx, block in enumerate(state_partition):
        for state in block:
            x.append(state)
            y.append(idx)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    f.fit(x, y)


def partition_improvement(partition, classifier, distance, k):

    new_partition = cp.deepcopy(partition)
    state_partition = get_state_partition(new_partition, classifier)

    for state_block in state_partition:

        flag = True

        while flag:

            flag = False

            for new_block in new_partition:

                tmp_new_partition = split(new_block, state_block, new_partition, classifier)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    flag = True
                    break

    induced_state_partition = induce_state_partition(new_partition, distance, k)
    train_f(induced_state_partition, classifier)

    return new_partition


def partition_iteration(partition, classifier, distance, k, max_steps=50):

    new_partition = partition_improvement(partition, classifier, distance, k)
    step = 0

    while partition != new_partition and step < max_steps:

        partition = new_partition
        new_partition = partition_improvement(partition, classifier, distance, k)

        step += 1

    return new_partition


def add_new_experience(state_action_partition, experience, classifier):

    d = {}

    for block in state_action_partition:
        sample = random.sample(block, 1)[0]
        key = (sample[2], classifier.predict(sample[3]))
        if key not in d:
            d[key] = []
        d[key] += block

    for state, action, reward, next_state, done in experience:
        key = (reward, classifier.predict(next_state))
        if key not in d:
            d[key] = []
        d[key].append((state, action, reward, next_state, done))

    new_state_action_partition = {frozenset(value) for value in d.values()}

    return new_state_action_partition


def reclassify_partition(state_action_partition, classifier):

    d = {}

    for block in state_action_partition:

        for state, action, reward, next_state, done in block:

            key = (reward, classifier.predict(next_state))
            if key not in d:
                d[key] = []
            d[key].append((state, action, reward, next_state, done))

    new_state_action_partition = {frozenset(value) for value in d.values()}

    return new_state_action_partition


def full_partition_iteration(gather_experience, classifier, distance, k, num_steps, reclassify=False):

    state_action_partition = set()

    for _ in range(num_steps):

        # collect experience
        experience = gather_experience()

        # add experience
        if len(state_action_partition) == 0:

            state_action_partition = {frozenset(experience)}

        else:

            if reclassify:
                state_action_partition.add(frozenset(experience))
                state_action_partition = reclassify_partition(state_action_partition, classifier)
            else:
                state_action_partition  = add_new_experience(state_action_partition, experience, classifier)

        # rearrange partition
        state_action_partition = partition_iteration(state_action_partition, classifier, distance, k)

    state_partition = get_state_partition(state_action_partition, classifier)

    return state_action_partition, state_partition