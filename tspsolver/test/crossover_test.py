import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.crossover import OnePointPMX


class SimplePopulationGeneratorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 100
        self._pop_size = 20

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size, self._distances)
        self._population = popGen.generate()

    def test_crossover(self):
        onept_pmx = OnePointPMX()
        new_pop = onept_pmx.crossover(self._population)

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        for row in new_pop:
            nose.tools.assert_equal(np.unique(row).size, row.size)
