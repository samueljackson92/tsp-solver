import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial import KDTree


class AbstractPopulationGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, population_size, **kwargs):
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

    def __init__(self, *args, **kwargs):
        AbstractPopulationGenerator.__init__(self, *args, **kwargs)
        self._random_proportion = kwargs.get('random_proportion', 0.5)

        if self._random_proportion < 0 or self._random_proportion > 1.0:
            raise ValueError("Probabilities must be in the range 0 <= x <= 1. Value was: %d"
                             % self._random_proportion)

    def generate(self, data):
        """ Generate a new random population of the given size. """
        num_points = data.shape[0]
        knn = KDTree(data, leafsize=10)
        population = []

        proportion_size = (1.0 - self._random_proportion) * self._population_size
        proportion_size = int(np.floor(proportion_size))

        # selection a proportion of
        for i in xrange(proportion_size):
            d, chromosome = knn.query(data[i], k=num_points, distance_upper_bound=20)
            population.append(chromosome)

        population = np.array(population)

        # generate random proportion of population
        random_gen = SimplePopulationGenerator(self._population_size - proportion_size)
        population = np.vstack((population, random_gen.generate(data)))

        return population
