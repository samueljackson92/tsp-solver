import unittest
import nose.tools
import logging

from ..tuning import GeneticAlgorithmParameterEstimation


class TuningTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.ERROR)

    def test_tuning(self):

        params = {
            "num_epochs": [10, 100, 1000],
            "num_elites": [3, 10, 15],
            "generator": ["SimplePopulationGenerator"],
            "generator_population_size": [30],
            "selector": ["TournamentSelection"],
            "selector_subset_size": [15],
            "selector_tournament_size": [3],
            "crossover": ["TwoPointPMX"],
            "crossover_pcross": [0.9],
            "mutator": ["SwapCityMutation"],
            "mutator_pmutate": [0.1]
        }

        tuner = GeneticAlgorithmParameterEstimation(num_datasets=3,
                                                    dataset_size=20)
        results = tuner.perform_grid_search(params)
        best_params = results.ix[results['fitness'].idxmin()]

        # check the parameters we were tuning are as expected.
        nose.tools.assert_equals(1000, best_params['num_epochs'])
        nose.tools.assert_equals(3, best_params['num_elites'])
        nose.tools.assert_equals((9, 12), results.shape)
