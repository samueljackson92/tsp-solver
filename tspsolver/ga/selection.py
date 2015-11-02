import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractSelection(object):
    __metaclass__ = ABCMeta

    def __init__(self, distance_matrix, subset_size):
        """Create a new selection technique.

        :param distance_matrix: the distance matrix of points in the dataset
        :param subset_size: the size of the subset of the population to use.
        """

        self._distance_matrix = distance_matrix
        self._population_size = distance_matrix.shape[0]
        self._subset_size = subset_size

        if self._subset_size > self._population_size:
            raise RuntimeError("Subset size cannot be larger than the \
                                total popualtion size.")

    @abstractmethod
    def selection(self, population):
        """ Choose a subset of a population to breed from.

        :param population: 2D array representing the population of solutions.
        :return: a subset of the population which are the fittest
        :rtype: ndarray
        """
        pass

    def fitness(self, population):
        """ Evaluate the fitness of a population

        :param population: 2D array representing the population of solutions.
        :return: array of total distances representing the fitness of each
                 solution
        :rtype: ndarray
        """
        return np.array([self._fitness_for_chromosome(x) for x in population])

    def _fitness_for_chromosome(self, chromosome):
        """Compute the fitness for a single chromosome

        :param chromosome: 1D array encoding to calcuate fitness for
        :return: 1D array of fitness estimates
        :rtype: ndarray
        """

        point_indices = zip(chromosome, np.roll(chromosome, 1))
        distances = np.array([self._distance_matrix[i, j]
                             for i, j in point_indices])
        return distances.sum()

    def _normalise_fitness(self, fitness):
        """Normalise the output of the fitness function.

        This allows them to be interepted as probabilities.

        :param fitness: 1D array of fitness estimates
        :return: 1D array of normalised fitness estimates
        :rtype: ndarray
        """
        total_fitness = fitness.sum()
        return np.array([f / total_fitness for f in fitness])


class RouletteWheelSelection(AbstractSelection):

    def _choose_subset(self, population, fit_prob):
        """Choose a random subset of te population weighted by their fitness
        probability.

        Chromosomes with a higher fitness (greater probability) are more likely
        to be selected than other individuals.

        :param population: 2D array representing the population of solutions.
        :param fit_prob: 1D array representing the normalised fitness.
        :return: a subset of the population
        :rtype: ndarray
        """
        pop_size = population.shape[0]
        indicies = np.arange(pop_size)
        idx = np.random.choice(indicies, size=self._subset_size, p=fit_prob)
        return population[idx]

    def selection(self, population):
        fitness = self.fitness(population)
        fit_prob = self._normalise_fitness(fitness)
        return self._choose_subset(population, fit_prob)
