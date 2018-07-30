import random


class PickEnv:

    STATES = [1, 2, 3, 4, 5]
    INITIAL_STATES = [1, 2, 3, 4]
    GOAL_STATE = 5
    ACTIONS = [1, 2, 3, 4]

    def __init__(self):

        self.state = None
        self.reset()

    def reset(self):

        self.state = random.choice(self.INITIAL_STATES)

    def step(self, action):

        if self.state == action:
            self.state = self.GOAL_STATE
            return 1, self.state
        else:
            return 0, self.state