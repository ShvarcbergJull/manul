from .abstract import Individ, Population
from .operators.base import ProgramRun

import ast
import numpy as np
from progress.bar import Bar
from numba.typed import Dict
from numba import njit, float64, int64
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


@njit
def chekkk(source_data, res, start_indexs):
    selects = np.zeros((len(source_data)))
    rem_edges = []
    while len(start_indexs) > 0:
        current_index = start_indexs.pop(0)
        selects[current_index] = 1
        if len(res[current_index]) == 0:
            continue
        kss = list(res[current_index].keys())[::-1]
        neigh_indxs = np.array([res[current_index][i] for i in kss])
        add_params = source_data[neigh_indxs]
        neighbours = source_data[current_index] - add_params

        for i, elem in enumerate(neigh_indxs):
            if selects[elem] == 1:
                continue
            check_this = source_data[elem]
            neigh_2 = check_this - add_params
            result = np.diag(np.dot(neighbours, neigh_2.T))
            if len(result[result < 0]) > 0:
                del res[current_index][kss[i]]
                rem_edges.append((current_index, elem))
            else:
                start_indexs.append(elem)
    
    return res, rem_edges

def forming_dict(graph, eds):
    # res = {}
    res = []
    for i in graph:
        res.append(Dict.empty(key_type=float64, value_type=int64))
        # res[i] = {}
        for k in graph[i]:
            res[i][eds[i][k]] = k

    return res

class DataStructureGraph(Individ):

    def __init__(self, data=None, labels=None, mode=1, n_neighbors=10, eps=0.5, graph_file: str = None):
        super().__init__()
        self.fullness = 0 # 0-100
        if data is None:
            return
        
        runner = ProgramRun()
        
        self.number_of_nodes = len(data)
        self.number_of_edges = 0
        
        if graph_file is not None:
                with open(graph_file, "r") as fl:
                    graph_data = fl.read()
                graph_data = ast.literal_eval(graph_data)
                self.load_graph(data, graph_data)
        elif mode:
            self.find_ED(eps, data)
            temp_edges = forming_dict(self.graph, self.matrix_connect)
            start_node_index = self.choosing_start_node()
            res, delete_edges = DataStructureGraph.check_visible_neigh(data, temp_edges, [start_node_index])
            runner.save_graph(res)
            self.local_remove(delete_edges)
        else:
            self.kernel = tp.tpgraph.Kernel(n_neighbors=n_neighbors, n_jobs=1, metric='cosine', fuzzy=True, verbose=True)
            self.kernel.fit(data)
            self.create_edges()


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

    def load_graph(self, data, res):
        eds = euclidean_distances(data, data)
        maxval = np.max(eds)
        self.matrix_connect = eds / maxval

        for i, edges in enumerate(res):
            self.graph[i] = edges
            self.number_of_edges += len(edges)

    def calc_fullness(self):
        self.fullness = (len(list(filter(lambda elem: elem == 0, self.laplassian.reshape(-1)))) / 2 * 100) // len(self.laplassian.reshape(-1))


    def copy(self):
        new_object = self.__class__()
        new_object.graph = deepcopy(self.graph)
        new_object.number_of_edges = self.number_of_edges
        new_object.number_of_nodes = self.number_of_nodes
        new_object.fullness = self.fullness
        new_object.matrix_connect = deepcopy(self.matrix_connect)

        return new_object
    
    def find_ED(self, eps, source_data):
        eds = euclidean_distances(source_data, source_data)
        maxval = np.max(eds)
        self.matrix_connect = eds / maxval

        for i in range(len(eds)):
            self.graph[i] = []
            for j in range(i, len(eds)):
                if eds[i][j] / maxval <= eps:
                    # self.graph.add_edge(i, j, weight=eds[i][j])
                    self.graph[i].append(j)
                    self.number_of_edges += 1


    def add_edge(self, from_node, to_node):
        self.graph[from_node].append(to_node)
        self.number_of_edges += 1

    def remove_edge(self, from_node, to_node):
        try:
            self.graph[from_node].remove(to_node)
        except:
            self.graph[to_node].remove(from_node)
        self.number_of_edges -= 1

    def create_edges(self):
        eds = euclidean_distances(self._source_data, self._source_data)
        maxval = np.max(eds)
        self.matrix_connect = eds / maxval
        lapl = self.kernel.L.todense()

        for i in range(len(eds)):
            self.graph[i] = []
            for j in range(i, len(eds)):
                if lapl[i][j] != 0:
                    # self.graph.add_edge(i, j, weight=eds[i][j])
                    self.graph[i].append(j)
                    self.number_of_edges += 1

    def choosing_start_node(self):
        choose_index = None
        for i, node in self.graph.items():
            try:
                if len(self.graph[choose_index]) < len(node):
                    choose_index = i
            except:
                choose_index = i

        return choose_index
    

    def local_remove(self, edges_list):
        # self.graph.remove_edges_from(edges_list)
        for edge in edges_list:
            try:
                self.graph[edge[0]].remove(edge[1])
            except:
                self.graph[edge[1]].remove(edge[0])
            self.number_of_edges -= 1

    
    def replace_subgraph(self, node: int, new_edges: dict):
        self.number_of_edges -= len(self.graph[node])
        self.graph[node] = []
        for elem in new_edges:
            self.add_edge(node, elem)


    @staticmethod
    @njit
    def check_visible_neigh(source_data, res, start_indexs):
        selects = np.zeros((len(source_data)))
        rem_edges = []
        while len(start_indexs) > 0:
            current_index = start_indexs.pop(0)
            selects[current_index] = 1
            if len(res[current_index]) == 0:
                continue
            kss = list(res[current_index].keys())[::-1]
            neigh_indxs = np.array([res[current_index][i] for i in kss])
            add_params = source_data[neigh_indxs]
            neighbours = source_data[current_index] - add_params

            for i, elem in enumerate(neigh_indxs):
                if selects[elem] == 1:
                    continue
                check_this = source_data[elem]
                neigh_2 = check_this - add_params
                result = np.diag(np.dot(neighbours, neigh_2.T))
                if len(result[result < 0]) > 0:
                    del res[current_index][kss[i]]
                    rem_edges.append((current_index, elem))
                else:
                    start_indexs.append(elem)
        
        return res, rem_edges

class PopulationGraph(Population):
    """
    Class with population of Graphs.
    """
    def __init__(self, structure: list = None,
                 iterations: int = 0):
        super().__init__(structure=structure)

        self.iterations = iterations
        self.type_ = "PopulationOfGraphs"
        self.change_fitness = []
        self.laplassian = None

    def _evolutionary_step(self, *args):
        # print(len(self.structure))
        self.apply_operator('FitnessPopulation')
        # print("Fitness")
        self.apply_operator('Elitism')
        # print("Elitism")
        self.apply_operator("RouletteWheelSelection")
        # print("Roulet")
        self.apply_operator("CrossoverPopulation")
        # print("Crossover")
        self.apply_operator("MutationPopulation")
        # print("Mutation")
        self.apply_operator("FilterPopulation")
        # print("Filter")

    def evolutionary(self, *args):
        print("INFO: create population")
        self.apply_operator('InitPopulation')
        bar = Bar('Evolution', max=self.iterations)
        bar.start()
        # поиск возможных вариантов где таргет закрепленный токен
        for n in range(self.iterations):
            # print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step()
            bar.next()
        bar.finish()

def _methods_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.change_all_fixes(False)
        return method(*args, **kwargs)
    return wrapper

class Draw:

    def __init__(self, graph) -> None:
        self.graph = graph

    def draw_lowd(self, nodes):
        edges=[]
        for edge in self.graph.edges:
            edges.append(edge.prev.new_params)
            edges.append(edge.next.new_params)
            edges.append([None for i in range(len(edge.prev.new_params))])
        
        edges = np.array(edges).T
        edge_trace = go.Scatter(x=edges[0], y=edges[1], line=dict(width=4, color='#888'), hoverinfo='none', mode='lines')
        
        nodes = np.array([node.new_params for node in self.graph.nodes]).T
        colors = np.array([node.color for node in self.graph.nodes])
        node_trace = go.Scatter(x=nodes[0], y=nodes[1], mode='markers', hoverinfo='text',
                                  marker=dict(
                                      showscale=True,
                                      # colorscale options
                                      # #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                      # #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                      # #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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
        
        return edge_trace, node_trace
    
    def draw_highd(self):
        edges=[]
        for edge in self.graph.edges:
            edges.append(self.graph.nodes[edge[0]]["params"])
            edges.append(self.graph.nodes[edge[1]]["params"])
            edges.append([None for i in range(len(self.graph.nodes[edge[0]]["params"]))])
        
        edges = np.array(edges).T
        edge_trace = go.Scatter3d(x=edges[0], y=edges[1], z=edges[2], line=dict(width=4, color='#888'), hoverinfo='none', mode='lines')
        
        nodes = np.array([self.graph.nodes[node]["params"] for node in self.graph.nodes]).T
        colors = np.array([self.graph.nodes[node]["label"] for node in self.graph.nodes])
        node_trace = go.Scatter3d(x=nodes[0], y=nodes[1], z=nodes[2], mode='markers', hoverinfo='text',
                                  marker=dict(
                                      showscale=True,
                                      # colorscale options
                                      # #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                      # #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                      # #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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
        
        return edge_trace, node_trace

    def draw_graph(self, mode=0, data=None):

        if mode:
            edge_trace, node_trace = self.draw_lowd(data)
        else:
            edge_trace, node_trace = self.draw_highd()       
        
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
        fig.show()

class TakeNN:

    def __init__(self, train_feature, train_target, dims, num_epochs, batch_size, model_settings=None):
        def baseline(dim):
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
                # nn.LogSoftmax(dim=1)
            )

            return baseline_model
        
        self.model_settings = {}

        if model_settings:
            self.model_settings = model_settings
        else:
            self.model_settings["model"] = baseline(dims)
            # self.model_settings["criterion"] = nn.BCELoss()
            self.model_settings["criterion"] = nn.L1Loss()
            self.model_settings['optimizer'] = Adam(self.model_settings['model'].parameters(), lr=1e-4, eps=1e-4)
        
        self.features = train_feature
        self.target = train_target
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.threshold = None
    
    def copy(self):
        new_object = self.__class__(self.features, self.target, 1, self.num_epochs, self.batch_size, model_settings=self.model_settings)
        new_object.threshold = deepcopy(self.threshold)
        # new_object.model_settings = deepcopy(self.model_settings)
        # new_object.features = deepcopy(self.features)
        # new_object.target = deepcopy(self.target)
        # new_object.num_epochs = deepcopy(self.num_epochs)
        # new_object.batch_size = deepcopy(self.batch_size)

        return new_object 

    def train(self, add_loss_func=None, graph=None, val=1):
        self.model_settings["model"].train()
        min_loss, t = np.inf, 0
        lmd = 1/((self.batch_size - 100) ** 2)
        epoch = 0
        end = False
        last_loss = None
        count_loss = 0
        # adding_loss = None
        # if add_loss_func:
        #     adding_loss = test_function(graph, train_features)
        while epoch < self.num_epochs and end == False:
            permutation = randperm(self.features.size()[0])
            loss_list = []
            for i in range(0, len(self.target), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                # print(indices)
                batch_x, target_y = self.features[indices], self.target[indices]
                target_y = target_y.to(fl64)
                self.model_settings["optimizer"].zero_grad()
                output = self.model_settings["model"](batch_x)
                loss = self.model_settings["criterion"](output, target_y.reshape_as(output))
                if add_loss_func:
                    add_loss = add_loss_func(graph, output.detach().numpy(), indices)
                    # add_loss = adding_loss[indices]
                    try:
                        loss += lmd * tensor(add_loss[0, 0])
                    except:
                        loss += lmd * tensor(add_loss)
                    # loss += lmd * torch.tensor(add_loss)
                # fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
                # gmeans = np.sqrt(tpr * (1-fpr))
                # ix = np.argmax(gmeans)
                # if not self.threshold:
                #     self.threshold = thresholds[ix]
                # else:
                #     self.threshold = np.mean([thresholds[ix], self.threshold])
                loss.backward()
                self.model_settings["optimizer"].step()
                # print(loss.item())
                loss_list.append(loss.item())
            # print(loss_list)
            loss_mean = np.mean(loss_list)

            if graph is not None:
                epoch += 1
                if t == 0:
                    last_loss = loss
                else:
                    if np.isclose(loss.detach().numpy(), last_loss.detach().numpy(), atol=1e-3):
                        count_loss += 1
                        # print("test")
                    last_loss = loss
                if count_loss >= 10:
                    end = True
            else:
                epoch += 1

            t += 1
            # print('Surface training t={}, loss={}'.format(t, loss_mean), count_loss)

        self.model_settings["model"].eval()

    
    def get_loss(self, add_loss_func=None, graph=None, val=1):
        output = self.model_settings["model"](self.features)
        target_y = self.target.to(fl64)
        # return_loss = roc_auc_score(target_y.reshape_as(output), output.detach().numpy())
        return_loss = mean_squared_error(target_y.reshape_as(output), output.detach().numpy())

        return 1/return_loss

    
    def get_current_loss(self, features, target, add_loss_func=None, graph=None):
        # lmd = 1/((len(features)) ** 2)
        output = self.model_settings["model"](features)
        output = output.detach().numpy()
        output = np.where(output > self.threshold, 1, 0)
        loss = f1_score(target.reshape(-1), output.reshape(-1), average='weighted')

        return loss
        # loss = self.model_settings["criterion"](output, target.reshape_as(output))
        # if add_loss_func:
        #     add_loss = add_loss_func(graph, output.detach().numpy())
        #     try:
        #         loss += lmd * tensor(add_loss[0, 0])
        #     except:
        #         loss += lmd * tensor(add_loss)