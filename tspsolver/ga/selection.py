import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractSelection(object):
    __metaclass__ = ABCMeta

    def __init__(self, distance_matrix, subset_size, **kwargs):
        """Create a new selection technique.

        :param distance_matrix: the distance matrix of points in the dataset
        :param subset_size: the size of the subset of the population to use.
        """

        self._distance_matrix = distance_matrix
        self._population_size = distance_matrix.shape[0]
        self._subset_size = subset_size

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

    def get_fitness(self):
        """Get the cached fitness of the most recent population.

        :return: the fitness for a population
        """
        return self._fitness

    def _fitness_for_chromosome(self, chromosome):
        """Compute the fitness for a single chromosome

        :param chromosome: 1D array encoding to calcuate fitness for
        :return: 1D array of fitness estimates
        :rtype: ndarray
        """
        point_indices = zip(chromosome, np.roll(chromosome, -1))
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
        fit_prob = np.array([(total_fitness - f) / total_fitness for f in fitness])
        p = fit_prob / fit_prob.sum()
        return p


class RouletteWheelSelection(AbstractSelection):
    """Roulette Wheel Selection

    Implements fitness propotionate or "roulette wheel" selection. Individuals
    are selected with probability that is directly proportional to their fitness
    """

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


class TournamentSelection(AbstractSelection):

    def __init__(self, *args, **kwargs):
        AbstractSelection.__init__(self, *args, **kwargs)
        self._tournament_size = kwargs.get('tournament_size', 3)
        self._winner_prob = kwargs.get('winner_prob', 1.0)

    def _find_tournament_chromosomes(self):
        """Select chromosomes for a torunament at random

        :return: 1darray of indicies for chromosomes the same size as the
            tournament size.
        """
        idx = np.random.choice(self._population_size,
                               size=self._tournament_size,
                               replace=False)
        return idx

    def _run_tournament(self, population, fitness):
        """Run a single tournament

        :param population: 2darray of chromosomes
        :param population: 1darray of fitness ratings for the population
        :return: the chromosome that won the tournament.
        """
        idx = self._find_tournament_chromosomes()
        winner_index = np.argmax(fitness[idx], axis=0)
        return population[winner_index]

    def selection(self, population):
        self._population_size = population.shape[0]
        self._fitness = self.fitness(population)

        new_pop = np.array([self._run_tournament(population, self._fitness)
                            for i in xrange(self._subset_size)])
        return new_pop
