from .base import GeneticOperatorIndivid, GeneticOperatorPopulation, apply_decorator
import numpy as np

from numba import njit

class MutationIndivid(GeneticOperatorIndivid):

    def __init__(self, params) -> None:
        super().__init__(params=params)
        self._check_params('mut_intensive')

    @staticmethod
    # @njit
    def collect_elements(node, neighbours):
        res = []
        for i in neighbours:
            res.append([node, i])
        
        return res



    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        mut_intensive = self.params['mut_intensive']
        fullness_individ = individ.fullness
        num_nodes = individ.number_of_nodes
        eds = individ.matrix_connect
        graph = individ.graph

        methods = np.random.choice(np.arange(2), size=mut_intensive, p=[fullness_individ / 100, 1 - (fullness_individ / 100)])
        for method in methods:
            if method:
                # добавление
                nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                while individ.laplassian[nodes[0]][nodes[1]] != 0:
                    nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                individ.add_edge(nodes[0], nodes[1])

            else:
                # удаление
                # edges = np.array([[int(elem[0]), int(elem[1]), elem[2]["weight"]] for elem in list(individ.graph.edges(data=True))])
                probability = []
                edges = []
                for key in graph:
                    probability.extend(eds[key, graph[key]])
                    edges.extend(MutationIndivid.collect_elements(key, graph[key]))
                probability = probability / np.sum(probability)
                edge_index = np.random.choice(np.arange(individ.number_of_edges), size=1, p=probability.astype(np.float64))[0]
                edge = edges[edge_index]
                individ.remove_edge(edge[0], edge[1])


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