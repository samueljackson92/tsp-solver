import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class Simulator:

    def __init__(self, generator, selector, crossover, mutator,  num_epochs=100, num_elites=2):
        self._generator = generator
        self._selector = selector
        self._crossover = crossover
        self._mutator = mutator
        self._num_elites = num_elites
        self._num_epochs = num_epochs

        self._average_fitness = []
        self._min_fitness = []
        self._max_fitness = []

    def evolve(self, distance_matrix):
        population = self.initilize_population(distance_matrix)

        for i in xrange(self._num_epochs):
            population = self._apply_genetic_operations(population,
                                                        distance_matrix)
            self._cache_epoch_performance(population)
            self._log_progress(i)

        self._log_final_output()
        return self._find_best_solution(population)

    def _cache_epoch_performance(self, population):
        fitness = self._selector.get_fitness()
        self._average_fitness.append(fitness.mean())
        self._min_fitness.append(fitness.min())
        self._max_fitness.append(fitness.max())

    def _log_progress(self, iteration):
        if iteration % 100 == 0:
            logger.info("------------------------------------------------")
            logger.info("Iteration %d" % iteration)
            logger.info("Current best solution: %d" % self._min_fitness[-1])
            logger.info("Best ever solution:    %d" % np.min(self._min_fitness))

    def _log_final_output(self):
        total_time = (time.time() - self._start_time)
        logger.info("------------------------------------------------")
        logger.info("FINISHED!")
        logger.info("Current best solution: %d" % self._min_fitness[-1])
        logger.info("Best ever solution:    %d" % np.min(self._min_fitness))
        logger.info("Execution Time:        %.2fs" % total_time)

    def _apply_genetic_operations(self, population, distance_matrix):
        subset = self.perform_selection(population, distance_matrix)
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

    def initilize_population(self, distance_matrix):
        logger.info("Begining simualtion...")
        self._start_time = time.time()
        return self._generator.generate(distance_matrix.shape[0])

    def perform_selection(self, population, distance_matrix):
        return self._selector.selection(population, distance_matrix)

    def perform_crossover(self, population):
        return self._crossover.crossover(population)

    def perform_mutation(self, population):
        return self._mutator.mutate(population)
