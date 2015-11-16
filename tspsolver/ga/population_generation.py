import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractPopulationGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, population_size):
        """Create a new population generator.

        :param population_size: the size of the population to generate
        :param distance_matrix: the distance matrix of points in the dataset
        """
        self._population_size = population_size

    @abstractmethod
    def generate(self):
        """ Generate a new random population of the given size. """
        pass


class SimplePopulationGenerator(AbstractPopulationGenerator):
    """Generate a population based on randomly shuffling 1D array of the
    indicies of every data point. This makes no attempt to consider any
    heuristic.
    """

    def generate(self, num_points):
        """ Generate a new random population of the given size. """
        population = np.array([np.random.permutation(num_points)
                               for _ in xrange(self._population_size)])
        return population
