import unittest
import nose.tools
import numpy as np

from ..generator.tsp_generator import TSPGenerator
from ..ga.simulator import Simulator


class SimulatorTest(unittest.TestCase):

    def setUp(self):
        self._num_points = 1000
        gen = TSPGenerator(self._num_points)
        self._data, self._distances = gen.generate()

    def test_evolve(self):
        sim = Simulator()
        solution = sim.evolve(num_epochs=1000)

        print solution
        assert False
