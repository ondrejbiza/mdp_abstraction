import copy as cp
import numpy as np


def split(block, states, partition, mdp):
    """
    Split a state-action block so that all state-actions in the smaller blocks have the same reward and the
    same probability of transitioning into a set of states given by the states argument.
    :param block:           A block to split.
    :param states:          A set of states for comparison of transition probabilities.
    :param partition:       A partition to modify.
    :param mdp:             An MDP.
    :return:                A modified partition.
    """

    partition = cp.deepcopy(partition)
    partition.remove(block)

    new_blocks = {}
    for state, action in block:

        reward = mdp.R[state]
        next_state = mdp.P[(state, action)]
        key = (reward, next_state in states)

        if key not in new_blocks.keys():
            new_blocks[key] = []

        new_blocks[key].append((state, action))

    for new_block in new_blocks.values():
        partition.add(frozenset(new_block))

    return partition


def get_state_partition(state_action_partition, f):
    """
    Get a state partition induced by state-action partition.
    :param state_action_partition:      A state-action partition.
    :return:                            A state partition.
    """

    state_partition = {}

    for state_action_block in state_action_partition:
        for state, _ in state_action_block:
            state_block = f.predict(state)
            if state_block not in state_partition:
                state_partition[state_block] = []
            state_partition[state_block].append(state)

    state_partition = {frozenset(value) for value in state_partition.values()}

    return state_partition


def induce_state_partition(state_action_partition, d, k):

    states = []
    blocks = []

    for block in state_action_partition:
        for state, action in block:
            states.append(state)
            blocks.append(block)

    dist_matrix = np.empty((len(states), len(states)), dtype=np.float32)

    for i, state_1 in enumerate(states):
        for j, state_2 in enumerate(states):
            if j > i:
                break
            dist_matrix[i, j] = d(state_1, state_2)

    state_partition = {}

    for idx, state in enumerate(states):
        tmp_blocks = set()
        nearest_neighbor_indexes = (-dist_matrix[idx]).argsort()[:k]
        for nearest_neighbor_index in nearest_neighbor_indexes:
            block = blocks[nearest_neighbor_index]
            tmp_blocks.add(block)
        tmp_blocks = frozenset(tmp_blocks)
        if tmp_blocks in state_partition:
            state_partition[tmp_blocks] = []
        state_partition[tmp_blocks].append(state)

    state_partition = {frozenset(value) for value in state_partition.values()}

    return state_partition


def train_f(state_partition, f):

    x = []
    y = []

    for idx, block in state_partition:
        for state, _ in block:
            x.append(state)
            y.append(idx)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    f.fit(x, y)


def partition_improvement(partition, mdp, f, d, k):
    """
    Perform one step of partition improvement.
    :param partition:       A partition.
    :param mdp:             An MDP.
    :return:                The same or a finer partition.
    """

    new_partition = cp.deepcopy(partition)
    state_partition = get_state_partition(new_partition, f)

    for state_block in state_partition:

        flag = True

        while flag:

            flag = False

            for new_block in new_partition:

                tmp_new_partition = split(new_block, state_block, new_partition, mdp)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    flag = True
                    break

    induced_state_partition = induce_state_partition(new_partition, d, k)
    train_f(induced_state_partition, f)

    return new_partition


def partition_iteration(mdp, f, d, k):
    """
    Iterate partition improvement until a fixed point is reached.
    :param mdp:         An MDP.
    :return:            The coarses state-action partition that is homomorphic to the original MDP.
    """

    all_pairs = mdp.P.keys()

    partition = {frozenset(all_pairs)}
    new_partition = partition_improvement(partition, mdp, f, d, k)

    while partition != new_partition:

        partition = new_partition
        new_partition = partition_improvement(partition, mdp, f, d, k)

    return new_partition