import random
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import environment


class StairEnv:

    # Define global vars

    # setup
    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def show(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError
