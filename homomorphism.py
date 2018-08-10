import copy as cp


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


def get_state_partition(state_action_partition):
    """
    Get a state partition induced by state-action partition.
    :param state_action_partition:      A state-action partition.
    :return:                            A state partition.
    """

    reversed_state_partition = {}
    ids = {block: idx for idx, block in enumerate(state_action_partition)}

    for block in state_action_partition:

        for state, _ in block:

            if state not in reversed_state_partition.keys():
                reversed_state_partition[state] = []
                reversed_state_partition[state].append(ids[block])

    state_partition = {}

    for key, value in reversed_state_partition.items():

        value = tuple(value)

        if value not in state_partition.keys():
            state_partition[value] = []

        state_partition[value].append(key)

    state_partition = {frozenset(value) for value in state_partition.values()}

    return state_partition


def partition_improvement(partition, mdp):
    """
    Perform one step of partition improvement.
    :param partition:       A partition.
    :param mdp:             An MDP.
    :return:                The same or a finer partition.
    """

    new_partition = cp.deepcopy(partition)
    state_partition = get_state_partition(new_partition)

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

    return new_partition


def partition_iteration(mdp):
    """
    Iterate partition improvement until a fixed point is reached.
    :param mdp:         An MDP.
    :return:            The coarses state-action partition that is homomorphic to the original MDP.
    """

    all_pairs = mdp.P.keys()

    partition = {frozenset(all_pairs)}
    new_partition = partition_improvement(partition, mdp)

    while partition != new_partition:

        partition = new_partition
        new_partition = partition_improvement(partition, mdp)

    return new_partition
