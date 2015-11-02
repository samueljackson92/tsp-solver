
class Simulator:

    def __init__(self, generator, selector, crossover, mutator):
        self._generator = generator
        self._selector = selector
        self._crossover = crossover
        self._mutator = mutator

    def initilize_population(self):
        pass

    def perform_selection(self):
        pass

    def perform_crossover(self):
        pass

    def perform_mutation(self):
        pass

    def evolve(self, num_epochs=100):
        pass
