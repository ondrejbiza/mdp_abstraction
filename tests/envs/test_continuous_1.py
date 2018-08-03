import unittest
import numpy as np
from envs.continuous_1 import ContinuousEnv1


class TestContinuous1(unittest.TestCase):

    def test_random_actions(self):

        env = ContinuousEnv1()

        for _ in range(10):

            for _ in range(5):

                env.reset()
                env.step(np.random.uniform(0, 2))