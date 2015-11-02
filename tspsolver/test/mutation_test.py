import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.mutation import SwapCityMutation


class SwapCityMutationTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        self._pop_size = 5

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size, self._distances)
        self._population = popGen.generate()

    def test_mutate(self):
        swap_city = SwapCityMutation(1.0)
        new_pop = swap_city.mutate(self._population.copy())

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        nose.tools.assert_true(np.any(new_pop != self._population))