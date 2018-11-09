import random
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
# import environment

class Polyomino:

    def__init__(self, shape=[1], location):
    """
    Polyomino class that allows you to represent with a shape vector and the
    location of the bottom left most block

    :param num_blocks       Size of Polyomino as an array
    :param location         Location of left most block
    """

    self.shape = shape
    self.location = location

class StairEnv:

    # Define global vars
    HAND_EMPTY = 0
    HAND_FULL = 1


    # setup
    def __init__(self, num_blocks=3, polyonimo_limit=1, only_horizontal=True, hall_length=3):
        """
        Create a staircase environment different sized blocks.
        :param num_blocks:          Number of blocks in the environment.
        :param polyonimo_limit      Max polyonimo size to be generated.
        :param only_horizontal      Whether or not to allow rotations of polyomino
        """
        self.num_blocks = num_blocks
        self.polyonimo_limit = polyonimo_limit
        self.only_horizontal = only_horizontal


    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def show(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError
