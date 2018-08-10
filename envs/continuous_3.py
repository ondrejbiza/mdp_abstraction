import numpy as np


class ContinuousEnv3:

    STATE_ACTION_MAP = np.array([
        [1, 2, 4],
        [0, 3, 4]
    ], dtype=np.int32)

    MINIMAL_STATE_ACTION_MAP = np.array([
        [1, 0, 3],
        [0, 2, 3]
    ], dtype=np.int32)

    STATE_MAP = np.array([0, 1, 2], dtype=np.int32)

    P = {
        0: 1,
        1: 0,
        2: 1,
        3: 2,
        4: 2
    }

    R = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 0
    }

    def __init__(self):

        self.state = None
        self.reset()

    def reset(self):

        self.state = np.random.uniform(0, 1)

    def step(self, action):

        state_action_block = self.get_state_action_block(self.state, action)

        next_state_block = self.P[state_action_block]
        next_state = np.random.uniform(next_state_block, next_state_block + 1)
        self.state = next_state

        reward = self.R[state_action_block]

        return reward, next_state, False

    def get_state_action_block(self, state, action):

        state_idx = int(np.floor(state))
        action_idx = int(np.floor(action))

        return self.STATE_ACTION_MAP[action_idx, state_idx]

    def get_state_action_block_minimal_map(self, state, action):

        state_idx = int(np.floor(state))
        action_idx = int(np.floor(action))

        return self.MINIMAL_STATE_ACTION_MAP[action_idx, state_idx]

    def get_state_block(self, state):

        state_idx = int(np.floor(state))
        return self.STATE_MAP[state_idx]

    @staticmethod
    def state_distance(state_1, state_2):

        return np.sum(np.abs(state_1 - state_2))