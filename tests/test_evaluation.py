import unittest
import numpy as np
import evaluation


class TestEvaluation(unittest.TestCase):

    class MockEnv:

        STATE_ACTION_MAP = np.array([
            [0, 1],
            [0, 2]
        ], dtype=np.int32)

        @classmethod
        def get_state_action_block(cls, state, action):

            state_idx = int(np.floor(state))
            action_idx = int(np.floor(action))

            return cls.STATE_ACTION_MAP[action_idx, state_idx]

    def test_overlap_total_match(self):

        block1 = [[0.1, 0.1], [0.4, 0.2]]
        block2 = [[1.5, 0.5], [1.3, 0.7]]
        block3 = [[1.7, 1.1], [1.9, 1.2]]

        partition = [block1, block2, block3]
        hits = evaluation.overlap(self.MockEnv, partition)

        self.assertEqual(hits, 6)

    def test_overlap_single_block(self):

        block1 = [[0.1, 0.1], [0.4, 1.2], [1.2, 0.1], [1.9, 1.9]]

        partition = [block1]
        hits = evaluation.overlap(self.MockEnv, partition)

        self.assertEqual(hits, 2)

    def test_overlap_two_blocks(self):

        block1 = [[0.1, 0.1], [0.4, 0.2]]
        block2 = [[1.5, 0.5], [1.3, 1.7]]

        partition = [block1, block2]
        hits = evaluation.overlap(self.MockEnv, partition)

        self.assertEqual(hits, 3)
