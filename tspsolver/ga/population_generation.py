import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial import KDTree


class AbstractPopulationGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, population_size):
        """Create a new population generator.

        :param population_size: the size of the population to generate
        :param distance_matrix: the distance matrix of points in the dataset
        """
        self._population_size = population_size

    @abstractmethod
    def generate(self, data):
        """ Generate a new random population of the given size. """
        pass


class SimplePopulationGenerator(AbstractPopulationGenerator):
    """Generate a population based on randomly shuffling 1D array of the
    indicies of every data point. This makes no attempt to consider any
    heuristic.
    """

    def generate(self, data):
        """ Generate a new random population of the given size. """
        population = np.array([np.random.permutation(data.shape[0])
                               for _ in xrange(self._population_size)])
        return population


class KNNPopulationGenerator(AbstractPopulationGenerator):
    """Generate a population based using the k nearest neighbours for each
    city.
    """

    def generate(self, data):
        """ Generate a new random population of the given size. """
        num_points = data.shape[0]
        knn = KDTree(data, leafsize=10)
        population = []
        for _ in xrange(self._population_size):
            index = np.random.randint(num_points)
            d, chromosome = knn.query(data[index], k=num_points, distance_upper_bound=20)
            population.append(chromosome)

        return np.array(population)
