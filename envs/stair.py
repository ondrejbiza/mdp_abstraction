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
    HAND_EMPTY = False
    HAND_FULL = True

    def __init__(self, env_extent, num_blocks=3, initial_state=np.array([])):

        """
        Create a staircase environment different sized blocks.
        Currently, stored as a hallway which depth values, so a solved environment
        might look like: [0, 0, 1, 2, 3], where a staircase is a sequence of increasing
        values.

        :parma env_extent           Length of the environment to place blocks
        :param num_blocks:          Number of blocks in the environment.
        :initial_state              Allows specification of initial env
        """

        self.num_blocks = num_blocks
        self.env_extent = env_extent
        self.num_actions = 2*env_extent - 1

        if (len(initial_state) > 1):
            assert initial_state[0].size == env_extent
            self.state = initial_state
        else:
            self.reset()

    def reset(self):

        # Clear the array first
        self.state = [np.zeros(self.env_extent), self.HAND_EMPTY]

        # For single blocks, randomly places them in the env
        for _ in range(self.num_blocks):
            i = np.random.choice(self.env_extent)
            self.state[0][i] += 1

    def step(self, action):
        '''
        :param      action: index of location to pick or place
        '''
        assert (action > -1) and (action <= self.num_actions)

        # Drop Block
        if action < self.env_extent:
            place_action = action

            # Check if hand full, otherwise nothing happens
            if self.state[1]:
                self.state[0][action] += 1; self.state[1] = self.HAND_EMPTY


        # Pick Up Block
        elif action < 2*self.env_extent:
            pick_action = action - self.env_extent

            # Check that hand is free and there are blocks to grab
            if (not self.state[1] and self.state[0][pick_action] > 0):
                self.state[0][pick_action] -= 1; self.state[1] = self.HAND_FULL

            # Or do nothing
            else:
                self.state[1] = self.HAND_EMPTY

        # Check if staircase to recieve reward
        if (self.is_stairs()):
            reward = 1
            done = True

        else:
            reward = 0
            done = False

        return reward, self.state, done

    def is_stairs(self):

        ind = np.argmax(self.state[0])

        # Loop through elements after argmax
        right_stair = True
        for i in range(ind + 1, self.state[0].size-1):
            if (self.state[0][i] != self.state[0][i+1] - 1):
                right_stair = False
                break

        # Loop through elements before argmax
        left_stair = True
        for i in reversed(range(1, ind)):
            if (self.state[0][i] != self.state[0][i-1] - 1):
                left_stair = False
                break

        return right_stair or left_stair

    def show(self):

        print(self.state[0])

    def get_state(self):

        return self.state
