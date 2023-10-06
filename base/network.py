from numba import njit, typed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import networkx as nx
import plotly.graph_objects as go

@njit
def distance(p_params, q_params):
    '''
    Searching the Euclidean distance between 2 points in the input matrix
    '''
    sm = 0
    for key in range(len(p_params)):
        dsq = (p_params[key] - q_params[key]) ** 2
        sm += dsq

    return np.sqrt(sm)

class Node:
    '''
    Class of point with its features, neigbours and edges (?)

    Arguments:
        name: (str) name of point
        params: (list) all features of the point
        neighbours: (list) list with nodes, that have edge with that point -> (?) может сосед расстояние (?)
        esges: (list) lenght of edge for each neighbours 

    Methods:
        ???
    '''

    def __init__(self, name_point:str, params:list, color:str=None) -> None:
        self.name = name_point
        self.params = params
        self.color = color
        self.neighbours = []
        self.edges = []
        self.count_neighbors = 0
        self.from_node = None
        self.min_distance = None
        self.visit = False
        self.transform = False
        self.select = False
        self.lenght = Node.find_norma(params)

        self.new_params = []
        self.raw_params = []

    @staticmethod
    @njit
    def find_norma(params):
        result = 0
        for param in params:
            result += np.power(param, 2)
        
        return np.sqrt(result)

    def __eq__(self, other):
        return self.name == other.name

    
    def __add_neighbour__(self, neigh_node) -> None:
        if neigh_node not in self.neighbours:
            self.neighbours.append(neigh_node)
            # self.edges.append(edge)

    def __delete_neighbour__(self, neigh_node) -> None:
        self.neighbours.remove(neigh_node)
        # self.edges.remove(edge)

    def get_data_for_pca(self):
        result = [self.params]
        colors = [self.color]
        for neigh in self.neighbours:
            result.append(neigh.params)
            colors.append(neigh.color)
            neigh.transform = True
        self.transform = True
        
        return result, colors
    
    def set_new_params(self, pca_params):
        self.new_params = pca_params[0]
        for i, params in enumerate(pca_params):
            if i == 0:
                continue
            self.neighbours[i - 1].new_params = params

class Graph(nx.Graph):

    def __init__(self, data, colors=None, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        for i, elem in enumerate(data):
            color = None
            if colors is not None:
                color = colors[i]                
            self.add_nodes_from([(i, {"name": i, "params": elem, "color": color, "select": False})])

        self.matrix_connect = np.zeros((self.number_of_nodes(), self.number_of_nodes()))
        self.find_ED(0.3)
        
    def find_ED(self, eps):
        N = self.number_of_nodes()
        edgesg = []
        maxval = None
        # print(graph.shape)
        for i in range(N):
            for j in range(i+1, N):
                val = distance(self.nodes[i]["params"],  self.nodes[j]["params"])
                edgesg.append((self.nodes[i], self.nodes[j], val))
                if not maxval:
                    maxval = val
                elif val > maxval:
                    maxval = val

        self.max_edge = maxval

        for i, edge in enumerate(edgesg):
            self.matrix_connect[edge[0]["name"]][edge[1]["name"]] = edge[2]/maxval
            self.matrix_connect[edge[0]["name"]][edge[1]["name"]] = edge[2]/maxval
            if edge[2]/maxval <= eps:
                self.add_edge(int(edge[0]["name"]), int(edge[1]["name"]), weight=edge[2])

    def search_nodes(self, n, choose_node):
        if n == 1:
            return [choose_node]
        distance_to_other = self.matrix_connect[int(choose_node.name)]
        index_node = np.argmax(distance_to_other)

        new_node = self.nodes[index_node]
        return [choose_node, new_node]
    
    def check_visible_neigh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours = sorted(self[current_node["name"]], key=lambda edge: edge[1]["weight"])

            for check_this_index in neighbours:
                check_this = self.nodes[check_this_index[0]]
                flag = False
                for neighbour in self.neighbors(current_node["name"]):
                    if check_this["name"] == neighbour["name"]:
                        continue
                    value = np.dot(current_node["params"] - neighbour["params"], check_this["params"] - neighbour["params"])

                    if value < 0:
                        flag = True
                        break

                if flag:
                    self.delete_edge([current_node, check_this])
                    current_node.__delete_neighbour__(check_this)
                    check_this.__delete_neighbour__(current_node)
            new_neighbours = filter(lambda x: not x.select, current_node.neighbours)
            start_nodes.extend(sorted(new_neighbours, key=lambda x: x.dist))

    def search_edge(self, nodes):
        try:
            distance = nx.get_edge_attributes(self, 'weight')[tuple(nodes)]
        except Exception as e:
            return None
        nodes.append(distance)
        return nodes

    def draw(self):
        edge_xyz = []
        for edge in self.edges:
            edge_xyz.append([self.nodes[int(edge[0])]["params"], self.nodes[int(edge[1])]["params"]])
        
        edge_xyz = np.array(edge_xyz)

        # 3d spring layout
        pos = nx.spring_layout(self, dim=3, seed=779)
        # Extract node and edge positions from the layout
        # node_xyz = np.array([pos[v] for v in sorted(draw_graph)])
        node_xyz = []
        for node in self.nodes.values():
            node_xyz.append(node["params"])
        node_xyz = np.array(node_xyz)
        # edge_xyz = np.array([(pos[u], pos[v]) for u, v in draw_graph.edges()])

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")


        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")


        _format_axes(ax)
        fig.tight_layout()
        plt.show()