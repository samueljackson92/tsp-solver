import logging
from ga.simulator import Simulator

logger = logging.getLogger(__name__)


class SimulatorLoader(object):

    def __init__(self, params):
        self._params = params

    def load(self):
        try:
            self._log_parameters(self._params)

            generator = self.load_component(self._params['generator'], 'population_generation')
            selector = self.load_component(self._params['selector'], 'selection')
            crossover = self.load_component(self._params['crossover'], 'crossover')
            mutator = self.load_component(self._params['mutator'], 'mutation')

            simulator = Simulator(
                generator=generator,
                selector=selector,
                crossover=crossover,
                mutator=mutator,
                **self._params['simulator']
            )
        except KeyError as e:
            logger.error("Could not find expected parameter: %s", str(e))
            raise e

        logger.info("Simulator successfully created")
        return simulator

    def load_component(self, params, module_name):
        module = __import__("tspsolver.ga.%s" % module_name, fromlist=[''])
        class_obj = getattr(module, params["name"])
        return class_obj(**params['params'])

    def _log_parameters(self, params):
        logger.info("Creating simulator with parameters:")
        logger.info("Generator:   %s" % params['generator']['name'])
        logger.info("Parameters:  %s" % params['generator']['params'])

        logger.info("Selector:    %s" % params['selector']['name'])
        logger.info("Parameters:  %s" % params['selector']['params'])

        logger.info("Crossover:   %s" % params['crossover']['name'])
        logger.info("Parameters:  %s" % params['crossover']['params'])

        logger.info("Crossover:   %s" % params['mutator']['name'])
        logger.info("Parameters:  %s" % params['mutator']['params'])
