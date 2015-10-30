import numpy as np


class TSPGenerator:
    def __init__(self, num_points, low=0, high=10):
        self._num_points = num_points
        self._low = low
        self._high = high

    def generate(self):
        return np.random.uniform(self._low, self._high,
                                 size=(self._num_points, 2))
