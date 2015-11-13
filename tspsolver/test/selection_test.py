import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.selection import RouletteWheelSelection, TournamentSelection


class RouletteWheelSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5
        self._subset_size = 3

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size, self._distances)
        self._population = popGen.generate()

    def test_evaluate_simple_fitness(self):
        distances = np.array([[1., 1.41421356],
                              [1.41421356, 1.]])

        population = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])

        selector = RouletteWheelSelection(distances, 1)
        fitness = selector.fitness(population)

        nose.tools.assert_equal(fitness.size, 4)

        expected = np.array([2., 2.82842712, 2.82842712, 2.])
        np.testing.assert_array_equal(fitness, expected)

    def test_evaluate_fitness(self):
        selector = RouletteWheelSelection(self._distances, self._subset_size)
        fitness = selector.fitness(self._population)
        nose.tools.assert_equal(fitness.size, self._pop_size)

    def test_selection(self):
        selector = RouletteWheelSelection(self._distances, self._subset_size)
        subset = selector.selection(self._population)
        exp_shape = (self._subset_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)


class TournamentSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5
        self._subset_size = 3

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size, self._distances)
        self._population = popGen.generate()

    def test_selection(self):
        selector = TournamentSelection(self._distances, self._subset_size)
        subset = selector.selection(self._population)

        exp_shape = (self._subset_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)
