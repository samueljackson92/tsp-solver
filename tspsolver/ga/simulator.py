import numpy as np


class Simulator:

    def __init__(self, generator, selector, crossover, mutator):
        self._generator = generator
        self._selector = selector
        self._crossover = crossover
        self._mutator = mutator

        self._average_fitness = []
        self._min_fitness = []
        self._max_fitness = []

    def evolve(self, num_epochs=100):
        population = self.initilize_population()

        for i in xrange(num_epochs):
            subset = self.perform_selection(population)
            new_population = self.perform_crossover(subset)
            new_population = self.perform_mutation(new_population)
            population = new_population

            if i % 1 == 0:
                fitness = self._selector.fitness(population)
                self._average_fitness.append(fitness.mean())
                self._min_fitness.append(fitness.min())
                self._max_fitness.append(fitness.max())
                best = population[np.argmin(fitness)]

        return best

    def get_averge_fitness(self):
        return self._average_fitness

    def get_max_fitness(self):
        return self._max_fitness

    def get_min_fitness(self):
        return self._min_fitness

    def initilize_population(self):
        return self._generator.generate()

    def perform_selection(self, population):
        return self._selector.selection(population)

    def perform_crossover(self, population):
        return self._crossover.crossover(population)

    def perform_mutation(self, population):
        return self._mutator.mutate(population)
