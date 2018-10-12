import copy as cp


def split(block, states, partition, mdp):
    """
    Perform the split operation assuming the MDP is deterministic.
    :param block:           A block in the partition to split.
    :param states:          States to compare against.
    :param partition:       Partition to modify.
    :param mdp:             An MDP.
    :return:                A new partition.
    """

    partition = cp.deepcopy(partition)
    partition.remove(block)

    new_blocks = {}
    for state in block:

        reward = mdp.R[state]
        key = [reward]

        for action in mdp.ACTIONS:
            key.append(mdp.P[state, action] in states)

        key = tuple(key)

        if key not in new_blocks.keys():
            new_blocks[key] = []

        new_blocks[key].append(state)

    for new_block in new_blocks.values():
        partition.add(frozenset(new_block))

    return partition


def partition_improvement(partition, mdp):
    """
    Improve the current partition (with respect to satisfying bisimulation).
    :param partition:       A partition.
    :param mdp:             An MDP.
    :return:                An improved partition.
    """

    new_partition = cp.deepcopy(partition)

    for block in partition:

        flag = True

        while flag:

            flag = False

            for new_block in new_partition:

                tmp_new_partition = split(new_block, block, new_partition, mdp)

                if new_partition != tmp_new_partition:

                    new_partition = tmp_new_partition

                    flag = True
                    break

    return new_partition


def partition_iteration(mdp):
    """
    Create the coarses partition that satisfies the bisimulation property.
    :param mdp:         An MDP.
    :return:            A partition.
    """

    partition = {frozenset(mdp.STATES)}
    new_partition = partition_improvement(partition, mdp)

    while partition != new_partition:

        partition = new_partition
        new_partition = partition_improvement(partition, mdp)

    return new_partition