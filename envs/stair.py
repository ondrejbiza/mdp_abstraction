import numpy as np
import matplotlib.pyplot as plt
# from gym import spaces
# import environment


# Not necessary for now, but if you want more
# complex block types, need a way to specify them
class Polyomino:

    def __init__(self, location, shape=[1]):

        """
        Polyomino class that allows you to represent with a shape vector and the
        location of the bottom left most block

        :param num_blocks       Size of Polyomino as an array
        :param location         Location of left most block
        """

        self.shape = shape
        self.location = location


class StairEnv:

    # Define fixed vars
    HAND_EMPTY = 0
    HAND_FULL = 1

    def __init__(self, env_extent, num_blocks=3, polyonimo_limit=1, only_horizontal=True, hall_length=3):

        """
        Create a staircase environment different sized blocks.
        Currently, stored as a hallway which depth values, so a solved environment
        might look like: [0, 0, 1, 2, 3], where a staircase is a sequence of increasing
        values.

        :parma env_extent           Length of the environment to place blocks
        :param num_blocks:          Number of blocks in the environment.
        :param polyonimo_limit      Max polyonimo size to be generated.
        :param only_horizontal      Whether or not to allow rotations of polyomino
        """

        self.num_blocks = num_blocks
        self.polyonimo_limit = polyonimo_limit
        self.only_horizontal = only_horizontal
        self.env_extent= env_extent

        self.state = [np.zeros(env_extent), StairEnv.HAND_EMPTY]

        self.reset()

    def reset(self):

        # For single blocks, randomly places them in the env
        for _ in range(self.num_blocks):
            i = np.random.choice(self.env_extent)
            self.state[0][i] += 1

        self.state[1] = self.HAND_EMPTY

    def step(self, action):
        '''
        :param      action: index of location to pick or place
        '''

        # If holding block, then drop it at desired location
        if self.state[1]:
            next_state = self.state[0][action] + 1
            self.state[1] = self.HAND_EMPTY

        # If not holding block, then pick it up at the desired index
        else:
            next_state = self.state[0][action] - 1 if (self.state > 0) else 0
            self.state[1] = self.HAND_FULL

        # Check if staircase to recieve reward
        if (self.is_stairs()):
            reward = 1
            done = True

        else:
            reward = 0
            done = False

        self.state = next_state

        return reward, next_state, done

    def is_stairs(self):

        ind = argmax(self.state[0])

        is_stair = True
        # Loop through elements after argmax
        for i in range(ind, len(self.state[0]-1):
            if (self.state[0][i] != self.state[0][i+1] - 1):
                is_stair = False
                break

        # Loop through elements before argmax
        for i in reversed(1, range(ind)):
            if (self.state[0][i] != self.state[0][i-1] - 1):
                is_stair = False
                break

        return False

    def show(self):

        print(self.state[0])

    def get_state(self):

        return self.state
