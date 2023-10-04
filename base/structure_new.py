from numba import njit, typed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import networkx as nx
import plotly.graph_objects as go

class Node:

    def __init__(self, name_point, params, color:str=None) -> None:
        self.name = name_point
        self.params = np.array(params)
        self.color = color
        self.neighbours = []
        self.from_node = None
        self.min_distance = None
        self.visit = False
        self.transform = False
        self.select = False
        self.lenght = Node.find_norma(params)

        self.new_params = []
        self.raw_params = []

    def __eq__(self, other):
        return self.name == other.name
    
    def __add_neighbour__(self, edge):
        if edge not in self.neighbours:
            self.neighbours.append(edge)
    
    @staticmethod
    @njit
    def find_norma(params):
        result = 0
        for param in params:
            result += np.power(param, 2)
        
        return np.sqrt(result)
    
    def get_data_for_pca(self):
        result = [self.params]
        colors = [self.color]
        for neigh in self.neighbours:
            result.append(neigh.params)
            colors.append(neigh.color)
            neigh.transform = True
        self.transform = True
        
        return result, colors
    
class Graph:


    def __init__(self, data, colors=None) -> None:
        points = []
        for i, exp in enumerate(data):
            if colors is not None:
                node = Node(str(i), exp, color=colors[i])
            else:
                node = Node(str(i), exp)
            points.append(node)

        self.avg = []
        self.var = []

        for i in range(len(data[0])):
            self.avg.append(np.average(data[:, i]))
            self.var.append(np.var(data[:, i]))

        self.nodes = points
        self.edges = []
        self.find_ED(0.3)

    
    def find_ED(self, eps):
        N = len(self.nodes)
        edgesg = []
        maxval = None
        # print(graph.shape)
        for i in range(N):
            for j in range(i+1, N):
                val = Edge((self.nodes[i], self.nodes[j]))
                edgesg.append(val)
                if not maxval:
                    maxval = val.distance
                elif val.distance > maxval:
                    maxval = val.distance

        self.max_edge = maxval

        for i, edge in enumerate(edgesg):
            if edge.distance/maxval <= eps:
                edge.add_neighbour_to_nodes()
                self.edges.append(edge)

    @staticmethod
    def _fitness_wrapper(params, *args):
        A, all_cos = args
        parametr = np.dot(A, params) - all_cos.T
        parameter = Node.find_norma(parametr)

        return  parameter ** 2
    
    def search_nodes(self, n, choose_node):
        result = [choose_node]
        if n == 1:
            return result
        distance_to_other = [neigh.distance for neigh in choose_node.neighbours]
        index_node = np.argmax(distance_to_other)

        new_node = choose_node.neighbours.get_neigh(choose_node)
        return [choose_node, new_node]
    
    def check_visible_neigh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node.select = True
            if len(current_node.neighbours) == 0:
                continue
            neighbours_edges = sorted(current_node.neighbours, key=lambda x: x.distance, reverse=True)
            neighbours = [neigh.get_neigh(current_node) for neigh in neighbours_edges]

            for i, check_this in enumerate(neighbours):
                flag = False
                if check_this.select:
                    continue
                nghts = np.array([neigh.params for neigh in neighbours if neigh != check_this])
                if len(nghts) == 0:
                    continue
                argument_0 = current_node.params - nghts
                argument_1 = check_this.params - nghts
                values = np.tensordot(argument_0, argument_1, axes=[1, 1])
                if len(values[values < 0]) > 0:
                    self.delete_edge(neighbours_edges[i])
            new_neighbours = filter(lambda x: not x.select, neighbours)
            start_nodes.extend(new_neighbours)

    def delete_edge(self, edge):
        edge.delete_neighbourhood()
        self.edges.remove(edge)

    def dijkstra(self, nodes):
        while len(nodes) > 0:
            node = nodes.pop(0)
            for next_node_edge in node.neighbours:
                next_node = next_node_edge.get_neigh(node)
                if next_node.min_distance is None or next_node.min_distance > node.min_distance + next_node_edge.distancee:
                    next_node.min_distance = node.min_distance + next_node_edge.distance
                    next_node.from_node = node
                if not next_node.visit:
                    nodes.append(next_node)
            node.visit = True

    def find_node_from(elf, nodes):
        max_trans = 0
        result_node = None

        for nn in nodes:
            your_neighs = list(filter(lambda x_node: x_node.get_neigh(nn).transform, nn.neighbours))
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
        for node in self.nodes:
            if node.from_node is not None and node.from_node == from_node and not node.transform:
                result_nodes.append(node)
        
        result_nodes = sorted(result_nodes, key=lambda x_node: x_node.min_distance)
        return result_nodes
    
    def find_raw_params(self, pca, center=None):
        for node in self.nodes:
            params = (node.params - self.avg) / self.var
            res = pca.transform([params])
            node.raw_params = res[0]

    def transform_part(self, nodes):
        for node in nodes:
            all_results = []
            rows = []

            a = node.params - node.from_node.params
            norm_of_a = Node.find_norma(node.raw_params - node.from_node.raw_params)
            for neigh_node in node.from_node.neighbours:
                if not neigh_node.transform:
                    continue
                # b = neigh_node.raw_params - node.from_node.raw_params
                b = neigh_node.params - node.from_node.params
                current_cos = np.dot(a, b) / (Node.find_norma(a) * Node.find_norma(b))
                all_results.append(current_cos)
                diff = neigh_node.new_params - node.from_node.new_params
                row = diff.T / (norm_of_a * Node.find_norma(diff))
                rows.append(row)
            x0 = node.from_node.new_params
            cons = ({'type': 'eq',
                'fun' : lambda x: Node.find_norma(x) - norm_of_a})
            res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(np.array(rows), np.array(all_results)), method='SLSQP', constraints=cons)
            node.new_params = res.x + node.from_node.new_params
            node.transform = True

    def transfrom_nodes(self, nodes, result, choosen_node):
        return_nodes = []
        while len(nodes) > 0:
            from_node, nodes = self.find_node_from(nodes)
            if from_node is None:
                nodes = []
                continue
            transform_nodes = self.find_all_next_nodes(from_node)
            if len(transform_nodes) == 0:
                continue
            self.transform_part(transform_nodes)

            nodes.extend(transform_nodes)
            return_nodes.extend(transform_nodes)

        return return_nodes
    
    def draw(self):
        draw_graph = nx.Graph()
        draw_graph.add_nodes_from([str(j) for j in range(len(self.nodes))])
        edge_xyz = []
        for edge in self.edges:
            draw_graph.add_edge(edge.nodes[0].name, edge.nodes[1].name, weight=edge.distance)
            edge_xyz.append([edge.nodes[0].params, edge.nodes[1].params])
        
        edge_xyz = np.array(edge_xyz)

        # 3d spring layout
        pos = nx.spring_layout(draw_graph, dim=3, seed=779)
        # Extract node and edge positions from the layout
        # node_xyz = np.array([pos[v] for v in sorted(draw_graph)])
        node_xyz = []
        for node in self.nodes:
            node_xyz.append(node.params)
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


class Edge:

    def __init__(self, points) -> None:
        self.nodes = tuple(points)
        self.distance = Edge.distance(self.nodes[0].params, self.nodes[1].params)

    def __eq__(self, __value: object) -> bool:
        if self.nodes == __value.nodes:
            return True
        return False

    @staticmethod
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
    
    def add_neighbour_to_nodes(self):
        self.nodes[0].__add_neighbour__(self)
        self.nodes[1].__add_neighbour__(self)

    def get_neigh(self, node):
        for ot_node in self.nodes:
            if ot_node != node:
                return ot_node
            
    def delete_neighbourhood(self):
        self.nodes[0].neighbours.remove(self)
        self.nodes[1].neighbours.remove(self)