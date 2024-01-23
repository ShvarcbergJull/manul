from .base import GeneticOperatorPopulation
from copy import deepcopy

import numpy as np

class InitPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        population.structure = []
        for _ in range(self.params['population_size']):
            new_individ = self.params['individ'].copy()
            if _ > 0:
                new_individ.apply_operator('MutationIndivid', mut_intensive=np.random.choice(np.arange(1, 6)))
            population.structure.append(new_individ)
        population.base_model = self.params['base_model'].copy()
        population.base_model.train()
        return population