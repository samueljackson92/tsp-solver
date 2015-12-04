import numpy as np
import logging
import time
from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class Simulator():

    def __init__(self, **params):
        """Create a new simulator.

        This will setup a genetic algorithm with the necessary genetic operators
        with their corresponding parameters ready to be evolved.

        :param generator: a generator object derived from AbstractPopulationGenerator
        :param selector: a selection operator derived from AbstractSelectionOperator
        :param crossover: a crossover operator derived from AbstractCrossoverOperator
        :param mutator: a mutation operator derived from AbstractMutationOperator
        :param num_epochs: the number of interations to run the simulation.
        :param num_elites: the number of elite chromosomes to carry over.
        """
        self.set_params(**params)
        self._average_fitness = []
        self._min_fitness = []
        self._max_fitness = []

    def evolve(self, data):
        """Evolve a solution to the TSP problem.

        :param distance_matrix: a distance matrix for the points in the dataset
        :return: chromosome representing the best solution found.
        :rtype: 1Darray
        """
        dm = distance_matrix(data, data)
        population = self.initilize_population(data)

        for i in xrange(self._num_epochs):
            population = self._apply_genetic_operations(population, dm)
            self._cache_epoch_performance(population)
            self._log_progress(i)

        self._log_final_output()
        return self._find_best_solution(population)

    def _cache_epoch_performance(self, population):
        """Store the min, max, and average of each generation

        This is useful for plotting to check convergance progression.

        :param population: the current population of solutions
        """
        fitness = self._selector.get_fitness()
        min_fitness = fitness.min()
        self._average_fitness.append(fitness.mean())
        self._min_fitness.append(min_fitness)
        self._max_fitness.append(fitness.max())

    def _log_progress(self, iteration):
        """Log the progress made every 100 iterations

        :param iteration: the current number of iterations
        """
        if iteration % 100 == 0:
            logger.info("------------------------------------------------")
            logger.info("Iteration %d" % iteration)
            logger.info("Current best solution: %d" % self._min_fitness[-1])
            logger.info("Best ever solution:    %d" % np.min(self._min_fitness))

    def _log_final_output(self):
        """Log a final output message on completion"""
        total_time = (time.time() - self._start_time)
        logger.info("------------------------------------------------")
        logger.info("FINISHED!")
        logger.info("Current best solution: %d" % self._min_fitness[-1])
        logger.info("Best ever solution:    %d" % np.min(self._min_fitness))
        logger.info("Execution Time:        %.2fs" % total_time)

    def _apply_genetic_operations(self, population, distance_matrix):
        """Apply each of the genetic operators

        Order of operations is to perform selection, crossover, mutation,
        and finally elitism.

        :param population: the current population of solutions
        :param distance_matrix: distance matrix for points in the dataset
        :return: a new population with the operators applied
        :rtype: 2Darray
        """
        subset = self.perform_selection(population, distance_matrix)
        new_population = self.perform_crossover(subset)
        new_population = self.perform_mutation(new_population)
        new_population = self._apply_elitism(population, new_population)
        return new_population

    def _apply_elitism(self, population, new_population):
        """Apply elitism to the new population

        Replace n new solutions with exact copies of the n best solutions from
        the previous generation. n is specified by the constructor parameter
        num_elites.

        :param population: the old population of solutions
        :param new_population: the new population of solutions
        :return: new population with elites carried over
        :rtype: 2Darray
        """
        fitness = self._selector.get_fitness()
        sorted_population = population[np.argsort(fitness)]
        new_population[:self._num_elites] = sorted_population[:self._num_elites]
        new_population = np.vstack((new_population, sorted_population[:self._num_elites]))
        return new_population

    def _find_best_solution(self, population):
        """Find the best solution in a population

        :param population: a 2Darray representing a population
        :return: the chromosome encoding the best solution
        :retype: 1Darray
        """
        fitness = self._selector.get_fitness()
        return population[np.argmin(fitness)]

    def get_averge_fitness(self):
        """Get the averge fitness of over all iterations

        :return: list of the average fitness over all iterations
        :rtype: 1Darray
        """
        return self._average_fitness

    def get_max_fitness(self):
        """Get the maximum fitness of over all iterations

        :return: list of the maximum fitness over all iterations
        :rtype: 1Darray
        """
        return self._max_fitness

    def get_min_fitness(self):
        """Get the minimum fitness of over all iterations

        :return: list of the minimum fitness over all iterations
        :rtype: 1Darray
        """
        return self._min_fitness

    def initilize_population(self, data):
        """Initilize the first population to a random set fo solutions

        :param distance_matrix: distance matrix for points in the dataset
        :return: a random population of solutions
        :rtype: 2Darray
        """
        logger.info("Beginning simulation...")
        self._start_time = time.time()
        return self._generator.generate(data)

    def perform_selection(self, population, distance_matrix):
        """Apply the selection operator to a population

        :param population: the population to apply the operator too
        :param distance_matrix: distance matrix for points in the dataset
        """
        return self._selector.selection(population, distance_matrix)

    def perform_crossover(self, population):
        """Apply the crossover operator to a population

        :param population: the population to apply the operator too
        """
        return self._crossover.crossover(population)

    def perform_mutation(self, population):
        """Apply the mutation operator to a population

        :param population: the population to apply the operator too
        """
        return self._mutator.mutate(population)

    def score(self, X):
        """Score function runs the genetic algorithm and returns the best
        fitness achieved.

        This is useful in parameter tuning to evaluate the settings of the GA.

        :param X: the data to run the GA with
        :return: an integer representing the best fitness achieved by this GA
        :rtype: Int
        """
        self.evolve(X)
        return np.min(self._min_fitness)

    def set_params(self, **parameters):
        """Set the parameters of the GA

        This takes a flat dictionary of keyword arguments for each of the
        components of the GA. Parameters must be named according to the
        convention '<component>_<param name>'. E.g. 'crossover_pcross'.

        This format is require for integration with scikit-learn tools, such as
        the ParameterGrid function.

        :param parameters: dictionary of parameters for the GA.
        """
        self._params = parameters
        # load params for simulator
        self._num_elites = parameters['num_elites']
        self._num_epochs = parameters['num_epochs']

        # load params for sub components
        self._generator = self.load_component('generator', self._params, 'population_generation')
        self._selector = self.load_component('selector', self._params, 'selection')
        self._crossover = self.load_component('crossover', self._params, 'crossover')
        self._mutator = self.load_component('mutator', self._params, 'mutation')

    def load_component(self, name, params, module_name):
        """Loads a sub component of the GA.

        This method dynamically sets up an instance of a class defined by the
        parameters passed to the system.

        :param name: the name of the type of component to load
        :param params: the parameter dictionary defining the algorithm
        :param module_name: the name of the python module to load from
        :return: a new instance of the component with it's paramters set
        :rtype: Object
        """
        module = __import__("tspsolver.ga.%s" % module_name, fromlist=[''])
        class_obj = getattr(module, self._params[name])

        params_for_class = {}
        for key, value in params.iteritems():
            if key.startswith(name + '_'):
                # find all parameters matching the prefix of this sub component
                # parameters for a sub component must begin with the component
                # name
                new_key = key.split(name + '_', 1)[1]
                params_for_class[new_key] = value

        return class_obj(**params_for_class)
