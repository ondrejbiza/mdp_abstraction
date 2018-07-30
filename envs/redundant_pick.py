STATES = [1, 2, 3, 4, 5, 11, 12, 13, 14]
INITIAL_STATES = [1, 2, 3, 4]
GOAL_STATE = 5
ACTIONS = [1, 2, 3, 4]

P = {}
R = {}

for state in STATES:

    if state == GOAL_STATE:
        R[state] = 1
    else:
        R[state] = 0

    for action in ACTIONS:
        if state == action or state == action + 10:
            P[(state, action)] = GOAL_STATE
        else:
            P[(state, action)] = state

assert len(P.keys()) == (4 * len(STATES))
assert len(R.keys()) == len(STATES)