import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractCrossoverOperator(object):
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
            if np.random.random() < self._pcross:
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


class OnePointPMX(AbstractCrossoverOperator):
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

        return child1, child2


class TwoPointPMX(AbstractCrossoverOperator):
    """Create a new population using one point PMX crossover. The pivot location
    of the split is determined uniformly at random.
    """

    def _crossover_for_chromosomes(self, x, y):
        pivot1 = np.random.randint(x.size/2)
        pivot2 = np.random.randint(x.size/2, x.size)
        child1 = y.copy()
        child2 = x.copy()

        for i in xrange(pivot1, pivot2):
            # set duplicate entry to be the one we're going to overwrite
            child1[child1 == x[i]] = child1[i]
            # crossover the gene
            child1[i] = x[i]

            child2[child2 == y[i]] = child2[i]
            child2[i] = y[i]
        return child1, child2


class OrderCrossover(AbstractCrossoverOperator):
        """Create a new population by keeping a subtour of the chromosome
        and then copying from the other parent.
        """

        def _crossover_for_chromosomes(self, x, y):
            # find subtours
            pivot1 = np.random.randint(x.size/2)
            pivot2 = np.random.randint(x.size/2, x.size)
            subtour1 = x[pivot1:pivot2]
            subtour2 = y[pivot1:pivot2]

            # copy subtours to children
            child1 = np.empty(x.size)
            child1.fill(-1)  # fill with invalid number
            child2 = np.empty(x.size)
            child2.fill(-1)  # fill with invalid number
            child1[pivot1:pivot2] = subtour1
            child2[pivot1:pivot2] = subtour2

            # replace missing parts from other chromosomes
            subtour_size = subtour1.size
            child1 = self._replace_from_parent(child1, y, pivot2, subtour_size)
            child2 = self._replace_from_parent(child2, x, pivot2, subtour_size)
            return child1, child2

        def _replace_from_parent(self, child, parent, pos, tour_size):
            for i in range(pos, pos + (parent.size-tour_size)):
                index = i % parent.size
                j = index
                while parent[j] in child:
                    j += 1
                    j = j % parent.size

                child[index] = parent[j]

            return child
