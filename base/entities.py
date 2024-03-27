from .abstract import Individ, Population
from .operators.base import ProgramRun

import ast
import numpy as np
import time
from progress.bar import Bar
from numba.typed import Dict
import numba.types as tp
from numba import njit, float64, int64, int32
from datetime import datetime
from copy import deepcopy
from functools import singledispatchmethod

import topo as tp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve
import plotly.graph_objects as go
import torch.nn as nn
from torch import randperm, tensor, mean, sqrt 
from torch.optim import Adam
from torch import float64 as fl64
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# temp_edges = None
# selects = None
# source_data = None


def draw(graph):
    edges=[]
    for edge in graph.structure:
        pos1 = graph.structure[edge]['orig_pos']
        for neig in graph.structure[edge]['neighbours']:
            edges.append(pos1)
            edges.append(graph.structure[neig]['orig_pos'])
            edges.append([None, None, None])
    
    edges = np.array(edges).T
    print(edges.shape)
    edge_trace = go.Scatter3d(x=edges[0], y=edges[1], z=edges[2], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    nodes = np.array([graph.structure[node]['orig_pos'] for node in graph.structure]).T
    colors = np.array([graph.structure[node]['marker'] for node in graph.structure])
    print(nodes.shape)
    node_trace = go.Scatter3d(x=nodes[0], y=nodes[1], z=nodes[2], mode='markers',
                                marker=dict(
                                    showscale=True,
                                    colorscale='YlGnBu',
                                    reversescale=True,
                                    color=colors,
                                    size=10,
                                colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                titleside='right'
                                ),
                                line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.write_html("data_struct.html")


class DataStructureGraph(Individ):
    """
    Class for Individ. Keeping data about graph and model with graph.

    Attributes
    ----------
    fullnes : int
        The percentage of completion of the graph.
    new_individ : bool
        If the value is True, fitness is counted for this individual in the epoch. 
    number_of_nodes : int
        Count of nodes in the graph.
    nuber_of_edges : int
        Count of edges in the graph.
    basis : list
        List with indexes of nodes for using in train.
    """

    def __init__(self, data=None, labels=None, mode=1, n_neighbors=10, eps=0.5, graph_file: str = None):
        """
        Args
        ----
        data : numpy.array
            values of features for each data
        labels : numpy.array
            target's values for each data
        n_neighbors : int
            count near neighbours, using for reduction nodes in graph
        eps : float
            epsilon neighborhood for each node, using for searching ending edges
        graph_file : str
            name of file with ready graph
        """
        super().__init__()
        self.fullness = 0 # 0-100
        self.new_individ = True
        if data is None:
            return
        
        runner = ProgramRun()
        
        self.number_of_nodes = len(data)
        self.number_of_edges = 0
        self.basis = np.arange(len(data))
        
        if graph_file is not None:
            self.find_ED(eps, data)
            self.name_gr = f"{runner.get_path()}/graph.txt"
            self.save_end_graph("rer")
        else:
            self.kernel = tp.tpgraph.Kernel(n_neighbors=n_neighbors, n_jobs=1, metric='cosine', fuzzy=True, verbose=True)
            self.kernel.fit(data)
            self.find_ED(eps, data)
            self.basis = self.searсh_basis(self.graph, source_data=data)
            del self.kernel
            self.number_of_nodes = len(self.basis)
            self.find_ED(eps, data[self.basis])
            time1 = time.time()
            res = self.check_visible(data[self.basis])
            print((time.time() - time1) * 1000)
            self.save_end_graph("base")
    
        self.draw_2d_projection(data[self.basis])
        self.calc_fullness()

    
    @property
    def laplassian(self):
        laplassian = np.zeros_like(self.matrix_connect)
        temp = 1 - self.matrix_connect
        for key in self.graph:
            laplassian[[key], [self.graph[key]]] = temp[[key], [self.graph[key]]]
        
        return laplassian

    def __eq__(self, __value: object) -> bool:
        return self.graph == __value.graph
    
    def copy(self):
        new_object = self.__class__()
        new_object.graph = deepcopy(self.graph)
        new_object.number_of_edges = self.number_of_edges
        new_object.number_of_nodes = self.number_of_nodes
        new_object.fullness = self.fullness
        new_object.matrix_connect = deepcopy(self.matrix_connect)
        try:
            new_object.model = self.model.copy()
        except:
            print("Step of init")

        return new_object

    def load_graph(self, data, res):
        """
        Method for load base graph to Individ

        Args:
        -----
        data : numpy.array
            matrix n * m, where n - number of nodes, m - number of features. 
            Keeping values of nodes by fields.
        res : list
            list size of n, where n - number of nodes.
            Each element is list with neighbour's indexes.  
        """
        eds = euclidean_distances(data, data)
        maxval = np.max(eds)
        self.matrix_connect = eds / maxval

        for i, edges in enumerate(res):
            self.graph[i] = edges
            self.number_of_edges += len(edges)

    def calc_fullness(self):
        """
        Method for calculation the percentage of completion of the graph.
        """
        self.fullness = (len(list(filter(lambda elem: elem == 0, self.laplassian.reshape(-1)))) / 2 * 100) // len(self.laplassian.reshape(-1))

    def save_end_graph(self, num):
        """
        Method save getting graph to file.

        Args:
        -----
        num : int
            Number that helps to distinguish graphs by each experiment.
        """
        runner = ProgramRun()
        res = []
        for i in range(self.number_of_nodes):
            try:
                val = self.graph[i]
            except:
                val = []
            
            res.append(val)
        runner.save_end_graph(res, name=f'graph_{num}.txt')


    def draw_2d_projection(self, data):
        import networkx as nx
        net_graph = nx.Graph()

        net_graph.add_nodes_from(np.arange(data.shape[0]))

        count = 0
        there_edge = np.zeros((data.shape[0], data.shape[0]))

        for i, val in enumerate(self.graph):
            for j in self.graph[val]:
                count += 1
                if i == j:
                    continue
                if there_edge[i][j] == 1 or there_edge[j][i] == 1:
                    continue
                net_graph.add_edge(i, j)

        my_object = IsolateGraph(data=data, colors=data, graph=net_graph)
        draw(my_object)
    
    def find_ED(self, eps, source_data):
        """
        Method for searching started edges between nodes.

        Args:
        -----
        eps : float
            Epsilon neighbothood for nodes.
        source_data : np.array
            matrix n * m, where n - number of nodes, m - number of features. 
            Keeping values of nodes by fields.
        """
        self.number_of_edges = 0
        self.graph = {} 
        eds = euclidean_distances(source_data, source_data)
        maxval = np.max(eds)
        self.matrix_connect = eds / maxval
        self.different = np.zeros((eds.shape[0], eds.shape[0], source_data.shape[1]))
        self.adjactive = np.zeros(eds.shape)
        k = 0

        if hasattr(self, 'kernel'):
            lapl = self.kernel.L.todense()

            for i in range(len(eds)):
                self.graph[i] = []
                for j in range(i, len(eds)):
                    if i == j:
                        continue
                    if lapl[i, j] != 0:
                        self.adjactive[i][j] = 1
                        self.adjactive[j][i] = 1
                        self.graph[i].append(j)
                        self.number_of_edges += 1
        else:
            for i in range(len(eds)):
                self.graph[i] = []
                self.different[i] = -1 * self.different[:, i]
                for j in range(i, len(eds)):
                    self.different[i][j] = source_data[i] - source_data[j]
                    if i == j:
                        continue
                    if eds[i][j] / maxval <= eps:
                        # self.graph.add_edge(i, j, weight=eds[i][j])
                        try:
                            if i in self.graph[j]:
                                print("th")
                                continue
                        except:
                            k = 1
                        self.adjactive[i][j] = 1
                        self.adjactive[j][i] = 1
                        self.graph[i].append(j)
                        self.number_of_edges += 1

            self.different = np.array(self.different)


    def add_edge(self, from_node, to_node):
        """
        Method for adding new edges.

        Args:
        -----
        from_node : int
            start node of the edge
        to_node : int
            end node of the edge
        """
        self.graph[from_node].append(to_node)
        self.number_of_edges += 1

    def remove_edge(self, from_node, to_node):
        """
        Method for removing edges.

        Args:
        -----
        from_node : int
            start node of the edge
        end_node : int
            end node of the edge
        """
        try:
            self.graph[from_node].remove(to_node)
        except:
            self.graph[to_node].remove(from_node)
        self.number_of_edges -= 1

    def choosing_start_node(self):
        """
        Method for searching node with maximum number of neighbours. 
        The found node will be used as the starting point when filtering neighbors.

        Returns:
        -------
        choose_node : int
            index of the found node
        """
        choose_index = None
        for i, node in self.graph.items():
            try:
                if len(self.graph[choose_index]) < len(node):
                    choose_index = i
            except:
                choose_index = i

        return choose_index
    

    def local_remove(self, edges_list):
        """
        Method for removing multiple edges from the list.

        Args:
        -----
        edges_list : list
            tuples with start and end nodes of edges.
        """
        for edge in edges_list:
            if edge[0] not in self.graph[edge[1]] and edge[1] not in self.graph[edge[0]]:
                continue
            try:
                self.graph[edge[0]].remove(edge[1])
            except:
                self.graph[edge[1]].remove(edge[0])
            self.number_of_edges -= 1
            
            
            if edge[0] in self.graph[edge[1]] or edge[1] in self.graph[edge[0]]:
                print("ux")


            if edge[0] in self.graph[edge[1]] or edge[1] in self.graph[edge[0]]:
                print("ux")

    
    def replace_subgraph(self, node: int, new_edges: list):
        """
        Method for replace some part of graph.

        Args:
        -----
        node : int
            index of the node whose connections with neighbours will be changed
        new_edges : list
            index of new neighbours for the node 
        """
        self.number_of_edges -= len(self.graph[node])
        self.graph[node] = []
        for elem in new_edges:
            self.add_edge(node, elem)

    
    def check_vn_part(self, source_data, node1, node2):
        """
        Method for check visible neighbours in new edge in graph (added using crossover/mutation)

        Args:
        -----
        source_data : numpy.array
            matrix n * m, where n - number of nodes, m - number of features. 
            Keeping values of nodes by fields.
        node1 : int
            one of nodes in the new edge
        node2 : int
            one of nodes in the new edge
        """
        general_neighbours = []
        del_list = []
        gr1 = self.graph[node1]
        gr2 = self.graph[node2]
        for neigh in gr1:
            if neigh in gr2:
                general_neighbours.append(neigh)

        data_neigh = source_data[general_neighbours]
        dif_n1 = source_data[node1] - data_neigh
        dif_n2 = source_data[node1] - source_data[node2]

        result = np.diag(np.dot(dif_n1, dif_n2.T))
        for i, res in enumerate(result[result < 0]):
            del_list.append((node1, general_neighbours[i]))
        
        dif_n1 = source_data[node2] - data_neigh
        dif_n2 = source_data[node2] - source_data[node1]

        result = np.diag(np.dot(dif_n1, dif_n2.T))
        for i, res in enumerate(result[result < 0]):
            del_list.append((node2, general_neighbours[i]))
        
        self.local_remove(del_list)
        
    def check_visible(self, data):
        """
        Method for filter the graph from unvisible neighbours.

        Args:
        -----
        data : numpy.array
            matrix n * m, where n - number of nodes, m - number of features. 
            Keeping values of nodes by fields.
        """
        # temp_edges = forming_dict(self.graph, self.matrix_connect)
        start_node_index = self.choosing_start_node()
        # res, delete_edges = DataStructureGraph.chicks(data, self.adjactive, self.matrix_connect, self.different, [start_node_index])
        # res, delete_edges = chekkk(data, temp_edges, [start_node_index])
        res, delete_edges = DataStructureGraph.chicks(data, self.adjactive, self.matrix_connect, [start_node_index])
        self.local_remove(delete_edges)

        return res
    
    @staticmethod
    @njit
    def chicks(source_data, adjactive, eds, start_nodes):
        selects = np.zeros((len(source_data)))
        rem_edges = []
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            selects[current_node] = 1
            if sum(adjactive[current_node]) == 0:
                continue
            neigh_indexs = np.where(adjactive[current_node] == 1)[0]
            args = np.argsort(eds[current_node, neigh_indexs])
            neigh_indexs = neigh_indexs[args[::-1]]

            add_params = source_data[neigh_indexs]
            neighbours = source_data[current_node] - add_params

            for i, elem in enumerate(neigh_indexs):
                if selects[elem] == 1:
                    continue
                check_this = source_data[elem]
                neigh_2 = check_this - add_params
                result = np.diag(np.dot(neighbours, neigh_2.T))
                if len(result[result < 0]) > 0:
                    adjactive[current_node][elem] = 0
                    adjactive[elem][current_node] = 0
                    rem_edges.append((current_node, elem))
                else:
                    start_nodes.append(elem)

        return adjactive, rem_edges
    
    @staticmethod
    def searсh_basis(graph, source_data):
        """
        Method for reducing nodes

        Args:
        -----
        source_data : numpy.array
            matrix n * m, where n - number of nodes, m - number of features. 
            Keeping values of nodes by fields.

        Returns:
        --------
        basis : list
            indexes of the nodes that we save in the graph
        """
        def forming_connect(graph):
            res = []
            for i in graph:
                res.append({'index': i, 'neighbours': [], 'stamp': False})
            for i in graph:
                for k in graph[i]:
                    res[k]['neighbours'].append(i)
                    res[i]['neighbours'].append(k)

            return res
        basis = []
        graph = forming_connect(graph)
        temp_graph = list(filter(lambda elem: not elem['stamp'], graph))
        while len(temp_graph) > 0:
            max_index = np.argmax([len(elem['neighbours']) for elem in temp_graph])
            use_index = temp_graph[max_index]['neighbours']
            use_index.append(temp_graph[max_index]['index'])
            average_values = np.average(source_data[use_index], axis=0)
            choose_point = use_index[0]
            for indx in use_index:
                if np.sqrt(((source_data[indx] - average_values) ** 2).sum()) < np.sqrt(((source_data[choose_point] - average_values) ** 2).sum()):
                    choose_point = indx
                graph[indx]['stamp'] = True
            basis.append(choose_point)
            # temp_graph = list(filter(lambda elem: not elem['stamp'], graph))
            temp_graph = list(filter(lambda elem: elem['index'] not in use_index, temp_graph))

        return basis



class PopulationGraph(Population):
    """
    Class with population of Graphs.

    Attributes
    ----------
    iterations : int
        number of epochs in population
    change_fitness : list
        the fitness values of th elit individ of each epoch 
    """
    def __init__(self, structure: list = None,
                 iterations: int = 0):
        super().__init__(structure=structure)

        self.iterations = iterations
        self.type_ = "PopulationOfGraphs"
        self.change_fitness = []

    def _evolutionary_step(self, *args):
        self.apply_operator('FitnessPopulation')
        self.apply_operator('Elitism')
        self.apply_operator("RouletteWheelSelection")
        self.apply_operator("CrossoverPopulation")
        self.apply_operator("MutationPopulation")
        self.apply_operator("FilterPopulation")

    def evolutionary(self, num, *args):
        print("INFO: create population")
        self.apply_operator('InitPopulation')
        bar = Bar('Evolution', max=self.iterations)
        bar.start()
        # поиск возможных вариантов где таргет закрепленный токен
        for n in range(self.iterations):
            self._evolutionary_step()
            bar.next()
        for individ in self.structure:
            if individ.elitism == True:
                individ.save_end_graph(num)
                break
        bar.finish()

def _methods_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.change_all_fixes(False)
        return method(*args, **kwargs)
    return wrapper

class TakeNN:
    """
    Class with neural network

    Attribites
    ----------
    problem : str
        type of problem, 2 tasks are being processed: 'class' and 'regres'
    model_settings : dict
        dictionary with all settings for NN (model, criterion, optimizer)
    """
    def __init__(self, train_feature, train_target, dims, num_epochs, batch_size, problem='class', model_settings=None):
        def baseline(dim):
            if problem == 'class':
                baseline_model = nn.Sequential(
                    nn.Linear(dim, 512, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(512, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 64, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(64, 1, dtype=fl64),
                    nn.Sigmoid()
                )
            elif problem == 'regres':
                baseline_model = nn.Sequential(
                    nn.Linear(dim, 512, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(512, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 64, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(64, 1, dtype=fl64)
                )

            return baseline_model
        
        self.model_settings = {}
        self.problem = problem

        if model_settings:
            self.model_settings = model_settings
        else:
            self.model_settings["model"] = baseline(dims)
            if problem == 'class':
                self.model_settings["criterion"] = nn.BCELoss()
            else:
                self.model_settings["criterion"] = nn.L1Loss()
            self.model_settings['optimizer'] = Adam(self.model_settings['model'].parameters(), lr=1e-4, eps=1e-4)
        
        self.features = train_feature
        self.target = train_target
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.threshold = None
    
    def copy(self):
        new_object = self.__class__(self.features, self.target, 1, self.num_epochs, self.batch_size, problem=self.problem, model_settings=self.model_settings)
        new_object.threshold = deepcopy(self.threshold)

        return new_object 

    def train(self, add_loss_func=None, graph=None, val=1):
        self.model_settings["model"].train()
        min_loss, t = np.inf, 0
        lmd = 1/((self.batch_size) ** 2)
        epoch = 0
        end = False
        last_loss = None
        count_loss = 0
        while epoch < self.num_epochs and end == False:
            permutation = randperm(self.features.size()[0])
            loss_list = []
            for i in range(0, len(self.target), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x, target_y = self.features[indices], self.target[indices]
                target_y = target_y.to(fl64)
                self.model_settings["optimizer"].zero_grad()
                output = self.model_settings["model"](batch_x)
                loss = self.model_settings["criterion"](output, target_y.reshape_as(output))
                if add_loss_func:
                    add_loss = add_loss_func(graph, output.detach().numpy(), indices)
                    try:
                        loss += lmd * tensor(add_loss[0, 0])
                    except:
                        loss += lmd * tensor(add_loss)
                if self.problem == 'class':
                    fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    if not self.threshold:
                        self.threshold = thresholds[ix]
                    else:
                        self.threshold = np.mean([thresholds[ix], self.threshold])
                loss.backward()
                self.model_settings["optimizer"].step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)

            if graph is not None:
                epoch += 1
                if t == 0:
                    last_loss = loss
                else:
                    if np.isclose(loss.detach().numpy(), last_loss.detach().numpy(), atol=1e-3):
                        count_loss += 1
                    last_loss = loss
                if count_loss >= 10:
                    end = True
            else:
                epoch += 1

            t += 1

        self.model_settings["model"].eval()

    
    def get_loss(self, add_loss_func=None, graph=None, val=1):
        output = self.model_settings["model"](self.features)
        target_y = self.target.to(fl64)
        nw_output = output.detach().numpy()
        nw_output = nw_output.round().astype("int64")
        if self.problem == 'class':
            return_loss = roc_auc_score(target_y.reshape_as(output), output.detach().numpy())
            return return_loss
        return_loss = mean_squared_error(target_y.reshape_as(output), nw_output)
        return 1/return_loss    

class IsolateGraph:
    def __init__(self, data, colors, graph):
        self.structure = {}
        self.avg = []
        self.var = []
        i = 0

        while i < len(data[0]):
            variance = np.var(data[:, i])
            if variance == 0:
                data = np.delete(data, i, 1)
                continue
            self.avg.append(np.average(data[:, i]))
            self.var.append(variance)
            i += 1

        self.eds = euclidean_distances(data, data)
        self.eds /= self.eds.max()
        self.neighbours = Dict.empty(key_type=int64, value_type=int32[:])
        for i in range(len(data)):
            neigh = []
            for edge in graph.edges(i):
                if edge[1] < i:
                    if i in self.structure[edge[1]]['neighbours']:
                        continue
                neigh.append(edge[1])
            self.structure[i] = {
                "name": i,
                "orig_pos": data[i],
                "neighbours": neigh,
                "marker": colors[i]
            }
            neighs = np.asarray([edge[1] for edge in graph.edges(i)])
            if len(neighs) > 0:
                self.neighbours[i] = neighs

    @staticmethod
    def _fitness_wrapper(params, *args):
        A, all_cos = args
        parametr = np.dot(A, params) - all_cos.T
        parameter = IsolateGraph.find_norma(parametr)

        return  parameter ** 2

    @staticmethod
    # @njit
    def get_started_point(graph_neigh):
        choose = None
        temp = None
        for i, neigh in enumerate(graph_neigh):
            if choose is None or len(neigh) > temp:
                choose = i
                temp = len(neigh)
        
        return choose
    
    @staticmethod
    @njit
    def fastik(neighbours, nodes, eds):
        min_distance = np.array([-1 for _ in range(len(neighbours))])
        from_node = np.array([0 for _ in range(len(neighbours))])
        visit = np.array([False for _ in range(len(neighbours))])
        min_distance[nodes[0]] = 0
        while len(nodes) > 0:
            node_index, nodes = nodes[0], nodes[1:]
            # node = self.structure[node_index]
            for next_node_index in neighbours[node_index]:
                # next_node = coord[next_node_index]
                # if next_node.min_distance is not None and next_node.min_distance == 0:
                #     continue
                their_edge = eds[node_index][next_node_index]
                if min_distance[next_node_index] == -1 or min_distance[next_node_index] > min_distance[node_index] + their_edge:
                    min_distance[next_node_index] = min_distance[node_index] + their_edge
                    from_node[next_node_index] = node_index
                
                if not visit[next_node_index]:
                    np.append(nodes, next_node_index)
            visit[node_index] = True

        return from_node, min_distance


    def dijkstra(self, nodes):
        from_nodes, min_d = IsolateGraph.fastik(self.neighbours, np.array(nodes), self.eds)
        for i, value in enumerate(from_nodes):
            self.structure[i]["from_node"] = value
            self.structure[i]["min_distance"] = min_d[i]
        # while len(nodes) > 0:
        #     node_index = nodes.pop(0)
        #     node = self.structure[node_index]
        #     for next_node_index in node["neighbours"]:
        #         next_node = self.structure[next_node_index]
        #         # if next_node.min_distance is not None and next_node.min_distance == 0:
        #         #     continue
        #         their_edge = self.eds[node_index][next_node_index]
        #         if next_node.get("min_distance", None) is None or next_node["min_distance"] > node["min_distance"] + their_edge:
        #             self.structure[next_node_index]["min_distance"] = node["min_distance"] + their_edge
        #             self.structure[next_node_index]["from_node"] = node_index
                
        #         if not next_node.get("visit", None):
        #             nodes.append(next_node_index)
        #     self.structure[node_index]["visit"] = True

    def get_data_for_pca(self, from_choose_node):
        result = [self.structure[from_choose_node]['orig_pos']]
        for neigh in self.structure[from_choose_node]["neighbours"]:
            result.append(self.structure[neigh]["orig_pos"])
            self.structure[neigh]["transform"] = True
        self.structure[from_choose_node]["transform"] = True
        
        return np.array(result)
    
    def set_new_params(self, from_choosen_node, pca_params):
        self.structure[from_choosen_node]["tr_pos"] = pca_params[0]
        index_neighbors = self.structure[from_choosen_node]['neighbours']
        for i, params in enumerate(pca_params):
            if i == 0:
                continue
            self.structure[index_neighbors[i - 1]]["tr_pos"] = params

    def find_raw_params(self, pca, center=None):
        for node_index in self.structure:
            node = self.structure[node_index]
            params = (node["orig_pos"] - self.avg) / self.var
            res = pca.transform([params])
            self.structure[node_index]["raw_pos"] = res[0]
            self.structure[node_index]["tr_pos"] = res[0]

    def transform_nodes(self, nodes):
        return_nodes = []
        while len(nodes) > 0:
            from_node, nodes = self.find_node_from(nodes)
            if from_node is None:
                nodes = []
                continue
            transform_nodes = self.find_all_next_nodes(from_node)
            if len(transform_nodes) == 0:
                continue
            # self.test_transform(transform_nodes)
            self.transform_part(transform_nodes)

            nodes.extend(transform_nodes)
            return_nodes.extend(transform_nodes)
        
        return return_nodes
    
    def find_node_from(self, nodes):
        max_trans = 0
        result_node = None

        for nn in nodes:
            your_neighs = list(filter(lambda x_node: self.structure[x_node].get("transform", False), self.structure[nn]['neighbours']))
            if len(your_neighs) > max_trans:
                max_trans = len(your_neighs)
                result_node = nn
        
        try:
            tr = nodes.remove(result_node)
        except Exception as e:
            print("there are transform all")
        return result_node, nodes
    
    def find_all_next_nodes(self, from_node):
        result_nodes = []
        for node_index in self.structure:
            node = self.structure[node_index]
            if node.get("from_node", None) is not None and node["from_node"] == from_node and not node.get("transform", None):
                result_nodes.append(node_index)
        
        result_nodes = sorted(result_nodes, key=lambda x_node: self.structure[x_node]["min_distance"])
        return result_nodes
    
    def transform_part(self, nodes):
        for node_index in nodes:
            all_results = []
            rows = []

            node = self.structure[node_index]
            from_node = self.structure[node["from_node"]]
            a = node["orig_pos"] - from_node["orig_pos"]
            norm_of_a = IsolateGraph.find_norma(node["raw_pos"] - from_node["raw_pos"])

            for neigh_node_index in from_node["neighbours"]:
                if not self.structure[neigh_node_index].get("transform", None):
                    continue

                b = self.structure[neigh_node_index]["orig_pos"] - from_node["orig_pos"]
                current_cos = np.dot(a, b) / (IsolateGraph.find_norma(a) * IsolateGraph.find_norma(b))
                all_results.append(current_cos)
                diff = self.structure[neigh_node_index]["tr_pos"] - from_node["tr_pos"]
                row = diff.T / (norm_of_a * IsolateGraph.find_norma(diff))
                rows.append(row)
            x0 = from_node["tr_pos"]
            cons = ({'type': 'eq',
                'fun' : lambda x: IsolateGraph.find_norma(x) - norm_of_a})
            res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(np.array(rows), np.array(all_results)), method='SLSQP', constraints=cons)
            self.structure[node_index]["tr_pos"] = res.x + from_node["tr_pos"]
            self.structure[node_index]["transform"] = True


    @staticmethod
    @njit
    def find_norma(params):
        result = 0
        for param in params:
            result += np.power(param, 2)
        
        return np.sqrt(result)