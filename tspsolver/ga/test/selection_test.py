import unittest
import nose.tools
import numpy as np
from scipy.spatial import distance_matrix

from tspsolver.tsp_generator import TSPGenerator
from ..population_generation import SimplePopulationGenerator
from ..selection import RouletteWheelSelection, TournamentSelection


class RouletteWheelSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._data)

    def test_evaluate_simple_fitness(self):
        distances = np.array([[1., 1.41421356],
                              [1.41421356, 1.]])

        population = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])

        selector = RouletteWheelSelection()
        selector._distance_matrix = distances
        fitness = selector.fitness(population)

        nose.tools.assert_equal(fitness.size, 4)

        expected = np.array([2., 2.82842712, 2.82842712, 2.])
        np.testing.assert_array_equal(fitness, expected)

    def test_evaluate_fitness(self):
        selector = RouletteWheelSelection()
        selector._distance_matrix = self._distances
        fitness = selector.fitness(self._population)
        nose.tools.assert_equal(fitness.size, self._pop_size)

    def test_selection(self):
        selector = RouletteWheelSelection()
        subset = selector.selection(self._population, self._distances)
        exp_shape = (self._pop_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)


class TournamentSelectionTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._data)

    def test_selection(self):
        selector = TournamentSelection()
        subset = selector.selection(self._population, self._distances)

        exp_shape = (self._pop_size, self._num_points)
        nose.tools.assert_equal(subset.shape, exp_shape)

    def test_run_tournament(self):
        selector = TournamentSelection()
        population = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 1, 4]])
        fitness = np.array([[200], [100], [300]])

        selector._tournament_size = 3
        selector._population_size = 3
        winner = selector._run_tournament(population, fitness)

        np.testing.assert_array_equal([[4, 3, 2, 1]], winner)
