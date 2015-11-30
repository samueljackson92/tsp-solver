import unittest
import nose.tools

from tspsolver.tsp_generator import TSPGenerator
from ..simulator import Simulator


class SimulatorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 30
        gen = TSPGenerator(self._num_points)
        self._data = gen.generate()

    def test_evolve(self):
        params = {
            "num_epochs": 100,
            "num_elites": 3,
            "generator": "SimplePopulationGenerator",
            "generator_population_size": 20,
            "selector": "TournamentSelection",
            "selector_subset_size": 15,
            "selector_tournament_size": 5,
            "crossover": "TwoPointPMX",
            "crossover_pcross": 0.6,
            "mutator": "SwapCityMutation",
            "mutator_pmutate": 0.2
        }

        sim = Simulator(**params)

        solution = sim.evolve(self._data)
        nose.tools.assert_equal(solution.shape, (self._num_points,))
