from .base import OperatorMap

from .initializers import InitPopulation

from .selectors import Elitism
from .selectors import RouletteWheelSelection
from .selectors import FilterPopulation

from .crossovers import CrossoverIndivid
from .crossovers import CrossoverPopulation

from .mutations import MutationIndivid
from .mutations import MutationPopulation

from .fitness import VarFitnessIndivid
from .fitness import FitnessPopulation


def create_operator_map(grid, individ, model, kwargs):
    mutation = kwargs['mutation']
    crossover = kwargs['crossover']
    population_size = kwargs['population']['size']
    fitness = kwargs['fitness']

    operatorsMap = OperatorMap()

    operatorsMap.InitPopulation = InitPopulation(
        params=dict(population_size=population_size,
                    individ=individ,
                    base_model=model))

    operatorsMap.Elitism = Elitism(
        params=dict(elitism=1, add_loss_function=fitness['add_loss_function'])
    )

    operatorsMap.RouletteWheelSelection = RouletteWheelSelection(
        params=dict(tournament_size=population_size, winners_size=int(0.5*population_size)+1)
    )

    operatorsMap.CrossoverPopulation = CrossoverPopulation(
        params=dict(crossover_size=int(0.4*population_size)+1)
    )

    operatorsMap.CrossoverIndivid = CrossoverIndivid(
        params=dict(cross_intensive=crossover['simple']['intensive'],
                    increase_prob=crossover['simple']['increase_prob'])
    )

    operatorsMap.MutationPopulation = MutationPopulation(
        params=dict(mutation_size=int(0.3*population_size)+1)
    )

    operatorsMap.MutationIndivid = MutationIndivid(
        params=dict(mut_intensive=mutation['simple']['intensive'],
                    increase_prob=mutation['simple']['increase_prob'],
                    source_data=grid.detach().numpy())
    )

    operatorsMap.VarFitnessIndivid = VarFitnessIndivid(
        params=dict(test_feature=fitness['test_feature'],
                    test_target=fitness['test_target'],
                    add_loss_function=fitness['add_loss_function'])
    )

    operatorsMap.FitnessPopulation = FitnessPopulation()

    operatorsMap.FilterPopulation = FilterPopulation(
        params=dict(population_size=population_size, source_data=grid.detach().numpy())
    )