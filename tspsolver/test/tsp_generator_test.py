import unittest
import nose.tools
import numpy as np
import scipy.stats as stats

from ..tsp_generator import TSPGenerator


class TSPGeneratorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 1000

    def test_create_dataset(self):
        gen = TSPGenerator(self._num_points)
        data, distances = gen.generate()

        nose.tools.assert_equal(data.shape, (self._num_points, 2))

        # check x axis is drawn from uniform distribution
        D, p_value = stats.kstest(data[:, 0], 'uniform', args=(0, 10))
        nose.tools.assert_greater(p_value, 0.05)

        # check y axis is drawn from uniform distribution
        D, p_value = stats.kstest(data[:, 1], 'uniform', args=(0, 10))
        nose.tools.assert_greater(p_value, 0.05)

        # check the distance matrix
        nose.tools.assert_equal(distances.shape, (self._num_points, self._num_points))
        nose.tools.assert_true(np.all(np.diag(distances) == 0))

    def test_create_dataset_with_bounds(self):
        # check lower bound param
        gen = TSPGenerator(self._num_points, low=5)
        data, distances = gen.generate()

        nose.tools.assert_equal(data.shape, (self._num_points, 2))
        nose.tools.assert_equal(data[data < 5].size, 0)

        # check upper bound param
        gen = TSPGenerator(self._num_points, high=5)
        data, distances = gen.generate()

        nose.tools.assert_equal(data.shape, (self._num_points, 2))
        nose.tools.assert_equal(data[data > 5].size, 0)

        # check both bounds together
        gen = TSPGenerator(self._num_points, low=5, high=15)
        data, distances = gen.generate()

        nose.tools.assert_equal(data.shape, (self._num_points, 2))
        nose.tools.assert_equal(data[data < 5].size, 0)
        nose.tools.assert_equal(data[data > 15].size, 0)
