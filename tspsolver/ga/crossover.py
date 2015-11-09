import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractCrossover(object):
    __metaclass__ = ABCMeta

    def __init__(self, pcross=0.6):
        self._pcross = pcross

    def crossover(self, population):
        """Peform crossover between pairs in a population

        :param population: 2darray of solutions to perform crossover on.
        :return: a new population with generated from the old one.
        :rtype: ndarray
        """

        pop = []
        for x, y in zip(population[::2], np.roll(population, -1, axis=0)[::2]):
            if self._pcross > np.random.random():
                c1, c2 = self._crossover_for_chromosomes(x, y)
                pop.append(c1)
                pop.append(c2)
            else:
                pop.append(x)
                pop.append(y)
        return np.array(pop)

    @abstractmethod
    def _crossover_for_chromosomes(self, x, y):
        """Peform crossover between a single pair in a population

        :param x: first parent to perform crossover on.
        :param y: second parent to perform crossover on.
        :return: a new solution with generated from the parents.
        :rtype: ndarray
        """
        pass

    def _parent_generator(self, population):
        """Randomly generate pairs of solutions to act as parents.

        This will iterate until a population of the same size has been
        generated.

        :param population: first parent to perform crossover on.
        :return: a tuple containg two solutions to act as parents.
        :rtype: (ndarray, ndarray)
        """
        pop_size = population.shape[0]

        for i in xrange(pop_size):
            x_idx = np.random.randint(pop_size)
            y_idx = np.random.randint(pop_size)
            yield population[x_idx], population[y_idx]


class OnePointPMX(AbstractCrossover):
    """Create a new population using one point PMX crossover. The pivot location
    of the split is determined uniformly at random.
    """

    def _crossover_for_chromosomes(self, x, y):
        pivot = np.random.randint(x.size)
        child1 = y.copy()
        child2 = x.copy()

        for i in xrange(pivot):
            # set duplicate entry to be the one we're going to overwrite
            child1[child1 == x[i]] = child1[i]
            # crossover the gene
            child1[i] = x[i]

            # set duplicate entry to be the one we're going to overwrite
            child2[child2 == y[i]] = child2[i]
            # crossover the gene
            child2[i] = y[i]

        return child1, child1
