import unittest
import nose.tools
import scipy.stats as stats

from ..generator.tsp_generator import TSPGenerator


class TSPGeneratorTest(unittest.TestCase):

    def test_create_dataset(self):
        gen = TSPGenerator(10)
        data = gen.generate()

        nose.tools.assert_equal(data.shape, (10, 2))

        # check x axis is drawn from uniform distribution
        D, p_value = stats.kstest(data[:, 0], 'uniform', args=(0,10))
        nose.tools.assert_greater(p_value, 0.05)

        # check y axis is drawn from uniform distribution
        D, p_value = stats.kstest(data[:, 1], 'uniform', args=(0,10))
        nose.tools.assert_greater(p_value, 0.05)

    def test_create_dataset_with_bounds(self):
        gen = TSPGenerator(10, low=5)
        data = gen.generate()

        nose.tools.assert_equal(data.shape, (10, 2))
        nose.tools.assert_equal(data[data<5].size, 0)

        gen = TSPGenerator(10, high=5)
        data = gen.generate()

        nose.tools.assert_equal(data.shape, (10, 2))
        nose.tools.assert_equal(data[data>5].size, 0)

        gen = TSPGenerator(10, low=5, high=15)
        data = gen.generate()

        nose.tools.assert_equal(data.shape, (10, 2))
        nose.tools.assert_equal(data[data<5].size, 0)
        nose.tools.assert_equal(data[data>15].size, 0)
