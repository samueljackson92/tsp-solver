import numpy as np
from sklearn.grid_search import ParameterGrid

from ga.simulator import Simulator
from tsp_generator import TSPGenerator
import pandas as pd


class GeneticAlgorithmParameterEstimation():

    def __init__(self, num_datasets, dataset_size):
        """Creates a new GA parameter tuner.

        This class can be used to tune the parameters of a GA but testing the
        performance across a range of different parameter settings.

        :param num_datasets: the number of datasets to generate
        :param dataset_size: the size of the dataset to generate
        """
        self._num_datasets = num_datasets
        self._generator = TSPGenerator(dataset_size)

    def perform_grid_search(self, params):
        """Perform a grid search over the range of parameters provided

        This will create a single set of parameters for each of the ranges
        specified in the params argument using scikit-learn's ParameterGrid
        function.

        Each set of sets of parameters is tested against multiple randomly
        generated datasets (of the same size). The average fitness achieved
        over all test datasets is taken as the measure of quality for the
        parameters.

        :param params: dictionary of ranges of parameters to be passed to ParameterGrid
        :return: dictionary of the best parameter settings found
        :rtype: dict
        """
        param_grid = list(ParameterGrid(params))
        self._param_fitness = []
        datasets = [self._generator.generate() for _ in range(self._num_datasets)]

        param_data = pd.DataFrame.from_dict(param_grid)

        for setting in param_grid:
            param_fitness = []
            for dataset in datasets:
                fitness = Simulator(**setting).score(dataset)
                param_fitness.append(fitness)

            median_fitness = np.median(param_fitness)# mean_fitness = float(mean_fitness) / self._num_datasets
            self._param_fitness.append(median_fitness)

        param_data['fitness'] = self._param_fitness
        return param_data

    def get_best_fitness(self):
        """ Get the average measure of fitness achieved over the datasets

        :return: the average measure of fitness
        :rtype: float
        """
        return np.min(self._param_fitness)
