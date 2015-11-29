import unittest
import nose.tools
import numpy as np
from scipy.spatial import distance_matrix

from ..tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.mutation import SwapCityMutation, SwapAdjacentCityMutation, DisplacementMutation


class SwapCityMutationTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances[0])

    def test_mutate(self):
        swap_city = SwapCityMutation(1.0)
        new_pop = swap_city.mutate(self._population.copy())

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        nose.tools.assert_true(np.any(new_pop != self._population))


class SwapAdjacentCityMutationTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances[0])

    def test_mutate(self):
        swap_city = SwapAdjacentCityMutation(1.0)
        new_pop = swap_city.mutate(self._population.copy())

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        nose.tools.assert_true(np.any(new_pop != self._population))


class DisplacementMutationTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()
        self._distances = distance_matrix(self._data, self._data)

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances[0])

    def test_mutate_single(self):
        mutator = DisplacementMutation(1.0)
        pop = np.array([[1, 2, 3, 4, 5, 6, 7]])
        new_pop = mutator.mutate(pop.copy())

        np.testing.assert_array_equal(pop[0], np.sort(new_pop[0]))

    def test_mutate(self):
        mutator = DisplacementMutation(1.0)
        new_pop = mutator.mutate(self._population.copy())

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        nose.tools.assert_true(np.any(new_pop != self._population))
