import numpy as np
from scipy.spatial import distance_matrix


class TSPGenerator:
    """
    Create a new TSP dataset generator object.
    """

    def __init__(self, num_points, low=0, high=10):
        """
        Create a new TSP dataset generator object.

        :param num_points: number of 2D data points to generate.
        :param low: lower bound of the range of coordinate space.
        :param high: upper bound of the range of coordinate space.
        """
        self._num_points = num_points
        self._low = low
        self._high = high

    def generate(self):
        """Generate a new TSP dataset.

        This will generate a uniformly random matrix of shape (N, 2)
        representing N 2D coordinates for cities and a distance matrix for each
        of the points in the dataset

        :return: 2D dataset of city coordinates and a n by n distance matrix
        :rtype: (ndarray, ndarray)
        """
        data = np.random.uniform(self._low, self._high, size=(self._num_points, 2))
        distances = distance_matrix(data, data)
        return data, distances
