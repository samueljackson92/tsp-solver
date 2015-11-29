import unittest
import nose.tools
import numpy as np
from scipy.spatial import distance_matrix

from ..tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.selection import RouletteWheelSelection, TournamentSelection


class RouletteWheelSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5
        self._subset_size = 3

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances.shape[0])

    def test_evaluate_simple_fitness(self):
        distances = np.array([[1., 1.41421356],
                              [1.41421356, 1.]])

        population = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])

        selector = RouletteWheelSelection(1)
        selector._distance_matrix = distances
        fitness = selector.fitness(population)

        nose.tools.assert_equal(fitness.size, 4)

        expected = np.array([2., 2.82842712, 2.82842712, 2.])
        np.testing.assert_array_equal(fitness, expected)

    def test_evaluate_fitness(self):
        selector = RouletteWheelSelection(self._subset_size)
        selector._distance_matrix = self._distances
        fitness = selector.fitness(self._population)
        nose.tools.assert_equal(fitness.size, self._pop_size)

    def test_selection(self):
        selector = RouletteWheelSelection(self._subset_size)
        subset = selector.selection(self._population, self._distances)
        exp_shape = (self._subset_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)


class TournamentSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5
        self._subset_size = 3

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances.shape[0])

    def test_selection(self):
        selector = TournamentSelection(self._subset_size)
        subset = selector.selection(self._population, self._distances)

        exp_shape = (self._subset_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)
