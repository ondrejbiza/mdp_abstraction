import copy as cp
import numpy as np
from mdp_abstraction.partition import Partition


def split(state_action_block, state_action_block_index, state_block, partition):
    """
    Split a state-action block with respect to a state block.
    :param state_action_block:      State-action block.
    :param state_block:             State block.
    :param partition:               State-action partition.
    :return:                        New state-action partition with possibly more blocks.
    """

    partition = cp.deepcopy(partition)
    partition.remove(state_action_block_index)

    new_blocks = {}
    for state, action, reward, next_state, done in state_action_block:

        # TODO: annotate state-action partition
        key = (reward, next_state in state_block)

        if key not in new_blocks.keys():
            new_blocks[key] = []

        new_blocks[key].append((state, action, reward, next_state, done))

    for new_block in new_blocks.values():
        partition.add(new_block)

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

    states = []
    for _, block in state_action_partition:
        for state, action, reward, next_state, done in block:
            states.append(state)
            states.append(next_state)

    state_partition = {}

    for state in states:
        actions = sample_actions(state)
        blocks = set()
        for action in actions:
            blocks.add(classifier.predict(state, action))
        key = frozenset(blocks)
        if key not in state_partition:
            state_partition[key] = []
        state_partition[key].append(state)

    tmp_state_partition = state_partition.values()
    state_partition = Partition()
    for block in tmp_state_partition:
        state_partition.add(block)

    return state_partition


def partition_improvement(partition, classifier, sample_actions, visualize_state_action_partition=None):
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

    for state_block_index, state_block in state_partition:

        flag = True

        while flag:

            flag = False

            for new_block_index, new_block in new_partition:

                tmp_new_partition = split(new_block, new_block_index, state_block, new_partition)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    if visualize_state_action_partition is not None:
                        print("split:")
                        visualize_state_action_partition(new_partition)

                    flag = True
                    break

    train_classifier(new_partition, classifier)

    return new_partition


def partition_iteration(partition, classifier, sample_actions, max_steps=2, visualize_state_action_partition=None):
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
        partition, classifier, sample_actions, visualize_state_action_partition=visualize_state_action_partition
    )
    step = 1

    while partition != new_partition and step < max_steps:

        partition = new_partition
        new_partition = partition_improvement(
            partition, classifier, sample_actions, visualize_state_action_partition=visualize_state_action_partition
        )

        step += 1

    return new_partition


def full_partition_iteration(gather_experience, classifier, sample_actions, num_steps,
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

    state_action_partition = Partition()
    all_experience = []

    for step in range(num_steps):

        # add experience
        all_experience += gather_experience()
        state_action_partition.add(all_experience)

        # visualize added experience
        if visualize_state_action_partition is not None:
            visualize_state_action_partition(state_action_partition)

        # rearrange partition
        state_action_partition = partition_iteration(
            state_action_partition, classifier, sample_actions,
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
