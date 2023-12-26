from .base import GeneticOperatorIndivid, GeneticOperatorPopulation, apply_decorator
import numpy as np

class MutationIndivid(GeneticOperatorIndivid):

    def __init__(self, params) -> None:
        super().__init__(params=params)
        self._check_params('mut_intensive')

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        mut_intensive = self.params['mut_intensive']

        methods = np.random.choice(np.arange(2), size=mut_intensive, p=[individ.fullness / 100, 1 - (individ.fullness / 100)])
        for method in methods:
            if method:
                # добавление
                nodes = np.random.choice(np.arange(individ.number_of_nodes()), size=2, replace=False)
                while individ.laplassian[nodes[0]][nodes[1]] != 0:
                    nodes = np.random.choice(np.arange(individ.number_of_nodes()), size=2, replace=False)
                new_weight = np.random.choice(np.linspace(np.min(individ.matrix_connect), np.max(individ.matrix_connect), individ.number_of_nodes()))
                individ.graph.add_edge(nodes[0], nodes[1], weight=new_weight)

                individ.laplassian[nodes[0]][nodes[1]] = 1 - individ.matrix_connect[nodes[0]][nodes[1]] / np.max(individ.matrix_connect)
                individ.laplassian[nodes[1]][nodes[0]] = 1 - individ.matrix_connect[nodes[0]][nodes[1]] / np.max(individ.matrix_connect)

            else:
                # удаление 
                edges = np.array([[int(elem[0]), int(elem[1]), elem[2]["weight"]] for elem in list(individ.graph.get_edges(data=True))])
                probability = edges[:, 2]
                probability = probability / probability.sum()
                edge_index = np.random.choice(np.arange(individ.number_of_edges()), size=1, p=probability.astype(np.float64))[0]
                edge = edges[edge_index]
                individ.graph.remove_edge(edge[0], edge[1])

                individ.laplassian[int(edge[0])][int(edge[1])] = 0
                individ.laplassian[int(edge[1])][int(edge[0])] = 0


class MutationPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mutation_size')

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        mutation_size = self.params['mutation_size']
        if mutation_size is None or mutation_size > len(selected_population):
            selected_individs = selected_population
        else:
            # assert mutation_size <= len(selected_population), "Mutations size must be less than population size"
            selected_individs = np.random.choice(selected_population, replace=False, size=mutation_size)

        for iter_ind, individ in enumerate(selected_individs):
            if individ.elitism:
                individ.elitism = False
                new_individ = individ.copy()
                new_individ.selected = False
                new_individ.elitism = True
                population.structure.append(new_individ)
            individ.apply_operator('MutationIndivid')
        return population