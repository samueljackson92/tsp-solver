import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.simulator import Simulator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.selection import RouletteWheelSelection
from ..ga.crossover import OnePointPMX
from ..ga.mutation import SwapCityMutation


class SimulatorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 10
        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

    def test_evolve(self):
        sim = Simulator(
            generator=SimplePopulationGenerator(100, self._distances),
            selector=RouletteWheelSelection(self._distances, 30),
            crossover=OnePointPMX(pcross=0.6),
            mutator=SwapCityMutation(p=0.05)
        )

        solution = sim.evolve(num_epochs=1000)
        nose.tools.assert_equal(solution.shape, (self._num_points,))
