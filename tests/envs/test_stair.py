import unittest
import numpy as np
from envs.stair import StairEnv

class TestStair(unittest.TestCase):
    SEED = 2384
    np.random.seed(SEED)


    def test_env_reset(self):
        env_extent = 10
        num_blocks = np.random.randint(0, 100)
        env = StairEnv(env_extent, num_blocks=num_blocks)

        for _ in range(10):
            env.reset()
            self.assertEqual(sum(env.state[0]), num_blocks)

    def test_is_stairs(self):
        left_stair = [np.array([2,1,0]), False]
        right_stair = [np.array([0,1,2]), False]

        env_extent = 3
        num_blocks = 3

        left_env = StairEnv(env_extent,num_blocks=num_blocks, initial_state=left_stair)
        right_env = StairEnv(env_extent,num_blocks=num_blocks, initial_state=right_stair)

        self.assertEqual(left_env.is_stairs(), True)
        self.assertEqual(right_env.is_stairs(), True)

    def test_env_step(self):
        env_extent = 2
        num_blocks = 1
        env = StairEnv(env_extent, num_blocks=1, initial_state=[np.array([1,0]), False])

        # Move block to other Location
        env.step(0)
        env.step(1)

        np.testing.assert_array_equal(env.state[0], np.array([0,1]))
        self.assertEqual(env.state[1], False)

if __name__ == '__main__':
    unittest.main()
