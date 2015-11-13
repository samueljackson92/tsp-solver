import numpy as np


class Simulator:

    def __init__(self, generator, selector, crossover, mutator, num_elites=2):
        self._generator = generator
        self._selector = selector
        self._crossover = crossover
        self._mutator = mutator
        self._num_elites = num_elites

        self._average_fitness = []
        self._min_fitness = []
        self._max_fitness = []

    def evolve(self, num_epochs=100):
        population = self.initilize_population()

        for i in xrange(num_epochs):
            population = self._apply_genetic_operations(population)
            self._cache_epoch_performance(population)

        return self._find_best_solution(population)

    def _cache_epoch_performance(self, population):
        fitness = self._selector.get_fitness()
        self._average_fitness.append(fitness.mean())
        self._min_fitness.append(fitness.min())
        self._max_fitness.append(fitness.max())

    def _apply_genetic_operations(self, population):
        subset = self.perform_selection(population)
        new_population = self.perform_crossover(subset)
        new_population = self.perform_mutation(new_population)
        new_population = self._apply_elitism(population, new_population)
        return new_population

    def _apply_elitism(self, population, new_population):
        fitness = self._selector.get_fitness()
        sorted_population = population[np.argsort(fitness)]
        new_population[:self._num_elites] = sorted_population[:self._num_elites]
        return new_population

    def _find_best_solution(self, population):
        fitness = self._selector.get_fitness()
        return population[np.argmin(fitness)]

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
