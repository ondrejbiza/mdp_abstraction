import copy as cp
import numpy as np


def split(state_action_block, state_block, partition, split_threshold):
    """
    Split a state-action block with respect to a state block.
    :param state_action_block:      State-action block.
    :param state_block:             State block.
    :param partition:               State-action partition.
    :return:                        New state-action partition with possibly more blocks.
    """

    partition = cp.deepcopy(partition)
    partition.remove(state_action_block)

    new_blocks = {}
    for state, action, reward, next_state, done in state_action_block:

        key = (reward, next_state in state_block)

        if key not in new_blocks.keys():
            new_blocks[key] = []

        new_blocks[key].append((state, action, reward, next_state, done))

    above_threshold = {}
    for key, value in new_blocks.items():
        if len(value) >= split_threshold:
            above_threshold[key] = True
        else:
            above_threshold[key] = False

    if not np.any(list(above_threshold.values())):
        # all blocks are smaller than  threshold, do nothing
        partition.add(state_action_block)
    else:
        for key, value in new_blocks.items():
            if not above_threshold[key]:
                # block is below threshold, add to closest block above threshold
                min_key = None
                min_distance = None

                for key2 in new_blocks.keys():
                    if above_threshold[key2]:
                        distance = edit_distance(key, key2)
                        if min_distance is None or min_distance > distance:
                            min_distance = distance
                            min_key = key2

                new_blocks[min_key] += new_blocks[key]

        # block is above threshold
        for key, value in new_blocks.items():
            if above_threshold[key]:
                assert len(value) >= split_threshold
                partition.add(frozenset(value))

    return partition


def train_classifier(state_action_partition, classifier):
    """
    Train a model to classify state-action pairs into blocks in state-action partition.
    :param state_action_partition:          State-action partition.
    :param classifier:                      Classifier.
    :return:                                None.
    """

    classifier.fit(state_action_partition)


def get_state_partition(state_action_partition, classifier, sample_actions):
    """
    Get a state partition from a state-action partition using a state-action classifier.
    :param state_action_partition:      State-action partition.
    :param classifier:                  Classifier.
    :param sample_actions:              Function that samples actions.
    :return:                            State partition.
    """

    states = set()
    for block in state_action_partition:
        for state, action, reward, next_state, done in block:
            states.add(state)
            states.add(next_state)

    state_partition = {}

    for state in states:
        actions = sample_actions(state)
        blocks = set(classifier.batch_predict(np.repeat(np.expand_dims(state, axis=0), len(actions), axis=0), actions))
        key = frozenset(blocks)
        if key not in state_partition:
            state_partition[key] = []
        state_partition[key].append(state)

    state_partition = set([frozenset(value) for value in state_partition.values()])
    return state_partition


def partition_improvement(partition, classifier, sample_actions, split_threshold,
                          visualize_state_action_partition=None):
    """
    Run a single step of partition improvement.
    :param partition:                               State-action partition.
    :param classifier:                              State-action classifier.
    :param sample_actions:                          Function for sampling actions.
    :param visualize_state_action_partition:        Visualize state-action partition.
    :return:                                        Improved state action partition.
    """

    new_partition = cp.deepcopy(partition)
    state_partition = get_state_partition(new_partition, classifier, sample_actions)

    for state_block in state_partition:

        flag = True

        while flag:

            flag = False

            for new_block in new_partition:

                tmp_new_partition = split(new_block, state_block, new_partition, split_threshold)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    if visualize_state_action_partition is not None:
                        print("split:")
                        visualize_state_action_partition(new_partition)

                    flag = True
                    break

    train_classifier(new_partition, classifier)

    return new_partition


def partition_iteration(partition, classifier, sample_actions, split_threshold, max_steps=2,
                        visualize_state_action_partition=None):
    """
    Run partition iteration.
    :param partition:                                   Initial partition.
    :param classifier:                                  State-action classifier.
    :param sample_actions:                              Sample actions function.
    :param max_steps:                                   Maximum number of partition iteration steps.
    :param visualize_state_action_partition:            Visualize state-action partition.
    :return:                                            New state-action partition.
    """

    new_partition = partition_improvement(
        partition, classifier, sample_actions, split_threshold,
        visualize_state_action_partition=visualize_state_action_partition
    )
    step = 1

    while partition != new_partition and step < max_steps:

        partition = new_partition
        new_partition = partition_improvement(
            partition, classifier, sample_actions, split_threshold,
            visualize_state_action_partition=visualize_state_action_partition
        )

        step += 1

    return new_partition


def full_partition_iteration(gather_experience, classifier, sample_actions, num_steps, split_threshold,
                             visualize_state_action_partition=None, visualize_state_partition=None,
                             max_iteration_steps=2):
    """
    Run the Full Partition Iteration algorithm.
    :param gather_experience:                       Gather experience function.
    :param classifier:                              State-action classifier.
    :param sample_actions:                          Sample actions function.
    :param num_steps:                               Number of steps.
    :param visualize_state_action_partition:        Visualize state-action partition.
    :param visualize_state_partition:               Visualize state partition.
    :param max_iteration_steps:                     Maximum number of partition improvement steps.
    :return:                                        State-action partition and state partition.
    """

    state_action_partition = set()
    all_experience = []

    for step in range(num_steps):

        # add experience
        all_experience += gather_experience()
        state_action_partition = {frozenset(all_experience)}

        # visualize added experience
        if visualize_state_action_partition is not None:
            visualize_state_action_partition(state_action_partition)

        # rearrange partition
        state_action_partition = partition_iteration(
            state_action_partition, classifier, sample_actions, split_threshold,
            visualize_state_action_partition=visualize_state_action_partition, max_steps=max_iteration_steps
        )

        # visualize rearranged partition
        if visualize_state_action_partition is not None:
            print("step {:d} state-action partition:".format(step + 1))
            visualize_state_action_partition(state_action_partition)

        # visualize state partition
        if visualize_state_partition is not None:
            print("step {:d} state partition:".format(step + 1))
            state_partition = get_state_partition(state_action_partition, classifier, sample_actions)
            visualize_state_partition(state_partition)

    state_partition = get_state_partition(state_action_partition, classifier, sample_actions)

    return state_action_partition, state_partition


def edit_distance(key1, key2):
    """
    https://www.geeksforgeeks.org/edit-distance-dp-5/
    """

    m = len(key1)
    n = len(key2)

    memory = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                memory[i][j] = j
            elif j == 0:
                memory[i][j] = i
            elif key1[i - 1] == key2[j - 1]:
                memory[i][j] = memory[i - 1][j - 1]
            else:
                memory[i][j] = 1 + min(memory[i][j - 1], memory[i - 1][j], memory[i - 1][j - 1])

    return memory[m][n]