import unittest
import nose.tools
import numpy as np

from ..tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator


class SimplePopulationGeneratorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 100
        self._pop_size = 20

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

    def test_generate_population(self):
        popGen = SimplePopulationGenerator(self._pop_size)
        population = popGen.generate(self._distances.shape[0])

        nose.tools.assert_equal(population.shape, (self._pop_size, self._num_points))
        unique_pop = find_unique_rows(population)
        nose.tools.assert_equal(unique_pop.size, population.size)


def find_unique_rows(matrix):
    """Find all of the unique rows in a matrix

    Code modified from:
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

    :param matrix: matrix which may contain identical rows
    :return: matrix with identical rows removed.
    :rtype: ndarray
    """

    tmp = np.ascontiguousarray(matrix).view(np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1])))
    _, idx = np.unique(tmp, return_index=True)
    return np.unique(tmp).view(matrix.dtype).reshape(-1, matrix.shape[1])
