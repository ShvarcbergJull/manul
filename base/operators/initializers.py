from .base import GeneticOperatorPopulation
from copy import deepcopy

class InitPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        population.structure = []
        for _ in range(self.params['population_size']):
            new_individ = self.params['individ'].copy()
            population.structure.append(new_individ)
        return population