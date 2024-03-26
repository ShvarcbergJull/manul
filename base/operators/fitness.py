import numpy as np

from .base import GeneticOperatorIndivid, GeneticOperatorPopulation

import cProfile, pstats, io

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__class__.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(datafn, 'w') as perf_file:
            perf_file.write(s.getvalue())
        return retval

    return wrapper

class VarFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('add_loss_function')

    def apply(self, individ, *args, **kwargs) -> None:
        individ.model = kwargs['base_model'].copy()
        individ.model.train(self.params["add_loss_function"], individ.laplassian)
        individ.fitness = individ.model.get_loss()


class FitnessPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            if not individ.new_individ:
                continue
            individ.apply_operator('VarFitnessIndivid', base_model=population.base_model)
            individ.calc_fullness()
            individ.new_individ = False
        return population


