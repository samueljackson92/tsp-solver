import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractMutation(object):
    __metaclass__ = ABCMeta

    def __init__(self, p=0.5):
        """Create a new mutation technique.

        :param p: the probability of the mutation occuring
        """
        self._mutation_prob = p

    @abstractmethod
    def mutate(self, population):
        """ Randomly mutate chromosomes in a population.

        :param population: 2D array representing the population of solutions.
        :return: mutated population
        :rtype: ndarray
        """
        pass


class SwapCityMutation(AbstractMutation):
    """Mutate indivudals in a population by randomly swapping two genes.
    """

    def mutate(self, population):
        for i, row in enumerate(population):
            if np.random.random() < self._mutation_prob:
                population[i] = self._swap_random_genes(row)
        return population

    def _swap_random_genes(self, chromosome):
        """Randomly swap two genes

        :param chromosome: 1D array representing a chromosome to mutate
        :return: 1D array representing the modified chromosome
        :rtype: ndarray
        """
        a = np.random.randint(chromosome.size)
        b = np.random.randint(chromosome.size)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome
