import numpy as np

from .base import GeneticOperatorIndivid, GeneticOperatorPopulation, apply_decorator

class CrossoverIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('cross_intensive', 'increase_prob')

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        cross_intensive = self.params['cross_intensive']
        increase_prob = self.params['increase_prob']

        ind1 = individ
        ind2 = kwargs['other_individ']

        probability = np.array([len(ind1.graph[i]) + len(ind2.graph[i]) for i in range(ind1.number_of_nodes())])
        probability = probability / probability.sum()
        start_node_index = np.random.choice(np.arange(ind1.number_of_nodes()), size=1, p=probability)[0]

        subgraph1 = dict(ind1.graph[start_node_index]).copy()
        subgraph2 = dict(ind2.graph[start_node_index]).copy()

        keys = list(subgraph1.keys())
        keys.extend(list(subgraph2.keys()))

        keys = np.unique(keys)

        temp_laplassian1 = ind1.laplassian.copy()
        temp_laplassian2 = ind2.laplassian.copy()
        for key in keys:
            temp_laplassian1[key, :] = ind2.laplassian[key, :]
            temp_laplassian1[:, key] = ind2.laplassian[:, key]
            temp_laplassian2[key, :] = ind1.laplassian[key, :]
            temp_laplassian2[:, key] = ind1.laplassian[:, key]
        
        ind1.laplassian = temp_laplassian1
        ind2.laplassian = temp_laplassian2

        ind1.replace_subgraph(start_node_index, subgraph2)
        ind2.replace_subgraph(start_node_index, subgraph1)



class CrossoverPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        crossover_size = self.params['crossover_size']
        if crossover_size is None or crossover_size > len(selected_population)//2:
            crossover_size = len(selected_population)//2
        selected_individs = [[selected_population[i], selected_population[j]] for i, j in np.random.choice(np.arange(len(selected_population)), replace=False, size=(crossover_size, 2))]
        for individ1, individ2 in selected_individs:
            n_ind_1 = individ1.copy()
            n_ind_2 = individ2.copy()
            n_ind_1.elitism = False
            n_ind_2.elitism = False
            population.structure.extend([n_ind_1, n_ind_2])
            n_ind_1.apply_operator('CrossoverIndivid', other_individ=n_ind_2)  # Параметры мутации заключены в операторе мутации с которым
        return population
        