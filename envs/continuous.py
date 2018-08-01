import numpy as np


map = np.array([
    [0, 0, 0, 4, 4, 4],
    [0, 0, 0, 4, 4, 4],
    [0, 0, 0, 0, 4, 4],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 3],
    [1, 2, 2, 3, 3, 3],
    [1, 2, 2, 3, 3, 3]

  ], dtype=np.int32)

P = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 4
}

R = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1
}


def get_block(state, action):

    state = int(np.floor(state))
    action = int(np.floor(action))
    return map[action, state]


def step(state, action):

    block = get_block(state, action)
    return R[block], P[block]


def dist(s1, s2):

    return np.abs(s1 - s2)