import unittest
import nose.tools
import numpy as np

from ..tsp_generator import TSPGenerator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.crossover import OnePointPMX, TwoPointPMX


class OnePointCrossoverTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 100
        self._pop_size = 20

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances[0])

    def test_crossover_for_chromosomes(self):
        onept_pmx = OnePointPMX()

        x = np.arange(10)
        y = x[::-1]

        c1, c2 = onept_pmx._crossover_for_chromosomes(x, y)

        nose.tools.assert_equal(np.unique(c1).size, c1.size)
        nose.tools.assert_equal(np.unique(c2).size, c2.size)

    def test_crossover(self):
        onept_pmx = OnePointPMX()
        new_pop = onept_pmx.crossover(self._population)

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        for row in new_pop:
            nose.tools.assert_equal(np.unique(row).size, row.size)


class TwoPointCrossoverTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 100
        self._pop_size = 20

        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

        popGen = SimplePopulationGenerator(self._pop_size)
        self._population = popGen.generate(self._distances[0])

    def test_crossover(self):
        twopt_pmx = TwoPointPMX()
        new_pop = twopt_pmx.crossover(self._population)

        nose.tools.assert_equal(new_pop.shape, self._population.shape)
        for row in new_pop:
            nose.tools.assert_equal(np.unique(row).size, row.size)

    def test_crossover_for_chromosomes(self):
        twopt_pmx = TwoPointPMX()

        x = np.arange(10)
        y = x[::-1]

        c1, c2 = twopt_pmx._crossover_for_chromosomes(x, y)

        nose.tools.assert_equal(np.unique(c1).size, c1.size)
        nose.tools.assert_equal(np.unique(c2).size, c2.size)
