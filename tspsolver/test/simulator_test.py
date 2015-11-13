import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.simulator import Simulator
from ..ga.population_generation import SimplePopulationGenerator
from ..ga.selection import RouletteWheelSelection, TournamentSelection
from ..ga.crossover import TwoPointPMX
from ..ga.mutation import SwapCityMutation, SwapAdjacentCityMutation


class SimulatorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 30
        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

    def test_evolve(self):
        sim = Simulator(
            generator=SimplePopulationGenerator(100, self._distances),
            selector=TournamentSelection(self._distances, 20, tournament_size=5),
            crossover=TwoPointPMX(pcross=0.6),
            mutator=SwapCityMutation(p=0.2)
        )

        solution = sim.evolve(num_epochs=1000)
        nose.tools.assert_equal(solution.shape, (self._num_points,))
