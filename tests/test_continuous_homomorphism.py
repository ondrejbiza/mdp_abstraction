import unittest
import numpy as np
import continuous_homomorphism


class MockClassifier:

    def __init__(self, d):

        self.d = d

    def predict(self, x):

        return self.d[x]


def manhattan_distance(s1, s2):

    return np.sum(np.abs(s1 - s2))


class TestContinuousHomomorphism(unittest.TestCase):

    def test_split(self):

        state_action_block = frozenset([(0, 0, 0, 0, False), (1, 0, 1, 0, False), (2, 0, 1, 0, False),
                                        (3, 0, 1, 1, False)])
        partition = {state_action_block}
        state_block = 1
        classifier = MockClassifier({0: 0, 1: 1})

        new_partition = continuous_homomorphism.split(state_action_block, state_block, partition, classifier)
        self.assertIn({(0, 0, 0, 0)}, new_partition)
        self.assertIn({(1, 0, 1, 0), (2, 0, 1, 0)}, new_partition)
        self.assertIn({(3, 0, 1, 1)}, new_partition)

    def test_induce_state_action_partition(self):

        block_1 = frozenset([(0.9, 0, 0, 0, False)])
        block_2 = frozenset([(1.1, 0, 1, 0, False)])
        block_3 = frozenset([(1.8, 0, 1, 0, False)])
        block_4 = frozenset([(2.2, 0, 1, 1, False)])

        state_action_parition = {block_1, block_2, block_3, block_4}

        state_partition = continuous_homomorphism.induce_state_partition(state_action_parition, manhattan_distance, 2)
        self.assertIn({0.9, 1.1}, state_partition)
        self.assertIn({1.8, 2.2}, state_partition)

    def test_get_state_partition(self):

        state_action_block = frozenset([(0, 0, 0, 0, False), (1, 0, 1, 0, False), (2, 0, 1, 0, False),
                                        (3, 0, 1, 1, False)])
        classifier = MockClassifier({0: 0, 1: 0, 2: 0, 3: 1})

        state_partition = continuous_homomorphism.get_state_partition({state_action_block}, classifier)
        self.assertIn({0, 1, 2}, state_partition)
        self.assertIn({3}, state_partition)
