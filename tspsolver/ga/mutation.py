import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractMutationOperator(object):
    __metaclass__ = ABCMeta

    def __init__(self, pmutate=0.5):
        """Create a new mutation technique.

        :param p: the probability of the mutation occuring
        """
        self._mutation_prob = pmutate

    def mutate(self, population):
        """ Randomly mutate chromosomes in a population.

        :param population: 2D array representing the population of solutions.
        :return: mutated population
        :rtype: ndarray
        """
        for i, row in enumerate(population):
            if np.random.random() < self._mutation_prob:
                population[i] = self._perform_mutation(row)
        return population

    @abstractmethod
    def _perform_mutation(self, chromosome):
        pass


class SwapCityMutation(AbstractMutationOperator):
    """Mutate indivudals in a population by randomly swapping two genes.
    """

    def _perform_mutation(self, chromosome):
        """Randomly swap two genes

        :param chromosome: 1D array representing a chromosome to mutate
        :return: 1D array representing the modified chromosome
        :rtype: ndarray
        """
        a = np.random.randint(chromosome.size)
        b = np.random.randint(chromosome.size)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome


class DisplacementMutation(AbstractMutationOperator):
    """Mutate indivudals in a population by randomly moving a subtour in the
    chromosome.
    """

    def _perform_mutation(self, chromosome):
        """Randomly displace a subtour of the chromosome

        :param chromosome: 1D array representing a chromosome to mutate
        :return: 1D array representing the modified chromosome
        :rtype: ndarray
        """
        # choose random subtour
        pivot1 = np.random.randint(chromosome.size/2)
        pivot2 = np.random.randint(chromosome.size/2, chromosome.size)
        subtour = chromosome[pivot1:pivot2]
        chromosome = np.concatenate((chromosome[:pivot1], chromosome[pivot2:]))

        # insert in random position
        pos = np.random.randint(chromosome.size)
        parts = (chromosome[:pos], subtour, chromosome[pos:])
        chromosome = np.concatenate(parts)
        return chromosome


class InversionMutation(AbstractMutationOperator):
    """Mutate indivudals in a population by randomly moving a subtour in the
    chromosome then reversing it.
    """

    def _perform_mutation(self, chromosome):
        """Randomly displace a (reversed) subtour of the chromosome

        :param chromosome: 1D array representing a chromosome to mutate
        :return: 1D array representing the modified chromosome
        :rtype: ndarray
        """
        # choose random subtour
        pivot1 = np.random.randint(chromosome.size/2)
        pivot2 = np.random.randint(chromosome.size/2, chromosome.size)
        subtour = chromosome[pivot1:pivot2]
        chromosome = np.concatenate((chromosome[:pivot1], chromosome[pivot2:]))

        # insert in random position and reverse
        pos = np.random.randint(chromosome.size)
        parts = (chromosome[:pos], subtour[::-1], chromosome[pos:])
        chromosome = np.concatenate(parts)
        return chromosome


class InsertionMutation(AbstractMutationOperator):
    """Mutate indivduals in a population by randomly removing and reinserting
    a gene in a different part of the chromosome.
    """

    def _perform_mutation(self, chromosome):
        """Randomly insert a gene in a different  of the chromosome

        :param chromosome: 1D array representing a chromosome to mutate
        :return: 1D array representing the modified chromosome
        :rtype: ndarray
        """
        # choose random gene
        pos = np.random.randint(chromosome.size)
        gene = chromosome[pos]
        chromosome = np.delete(chromosome, pos)

        # insert in random position
        pos = np.random.randint(chromosome.size)
        parts = (chromosome[:pos], np.array([gene]), chromosome[pos:])
        chromosome = np.concatenate(parts)
        return chromosome
