from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import networkx as nx

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

    def __init__(self, name_point:str, params:list) -> None:
        self.name = name_point
        self.params = params
        self.neighbours = []
        self.edges = []
        self.count_neighbors = 0
        self.from_node = None
        self.min_distance = None
        self.visit = False
        self.transform = False
        self.lenght = Node.find_norma(params)

        self.new_params = []

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

    def __delete_neighbour__(self, neigh_node) -> None:
        self.neighbours.remove(neigh_node)

    def get_data_for_pca(self):
        result = [self.params]
        for neigh in self.neighbours:
            result.append(neigh.params)
            neigh.transform = True
        self.transform = True
        
        return result
    
    def set_new_params(self, pca_params):
        self.new_params = pca_params[0]
        for i, params in enumerate(pca_params):
            if i == 0:
                continue
            self.neighbours[i - 1].new_params = params



class Graph:
    '''
    Class of graph with points

    Arguments:
        data

    Methods:
        ....
    '''

    def __init__(self, data) -> None:
        '''
        data must have size (n, m), where n - count experiments and m - count features of point
        '''

        points = []
        for i, exp in enumerate(data):
            node = Node(str(i), exp)
            points.append(node)

        self.nodes = points
        self.edges = []
        self.find_ED(0.3)

    @staticmethod
    def _fitness_wrapper(params, *args):
        A, all_cos = args
        parametr = np.dot(A, params) - all_cos
        parameter = Node.find_norma(parametr)

        return  parameter ** 2


    def get_names(self, node):
        names = node.name
        for n_node in self.nodes:
            names.append(n_node.name)

        return names
    
    def check_visible_neigh(self):
        for node in self.nodes:
            del_list = []
            for neigh_node in node.neighbours:
                for neigh_node2 in node.neighbours:
                    if neigh_node == neigh_node2:
                        continue

                    if neigh_node not in node.neighbours or neigh_node2 not in node.neighbours:
                        continue
                    
                    value = np.dot(node.params - neigh_node2.params, neigh_node.params - neigh_node2.params)
                    if value < 0:
                        del_neigh = self.near_neigh(node, [neigh_node, neigh_node2])
                        self.delete_edge([node, del_neigh])
                        node.__delete_neighbour__(del_neigh)
                        del_neigh.__delete_neighbour__(node)
                        # self.delete_edge([node, neigh_node2])
                        # node.__delete_neighbour__(neigh_node2)
                        # neigh_node2.__delete_neighbour__(node)

    def near_neigh(self, node, neighs):
        edge0 = self.search_edge([node, neighs[0]])
        edge1 = self.search_edge([node, neighs[1]])

        if edge0.distance < edge1.distance:
            return neighs[1]
        else:
            return neighs[0]

    def delete_edge(self, nodes):
        for edge in self.edges:
            if edge.prev == nodes[0] and edge.next == nodes[1]:
                self.edges.remove(edge)
                return True
            if edge.prev == nodes[1] and edge.next == nodes[0]:
                self.edges.remove(edge)
                return True
        return False
    
    def search_edge(self, nodes):
        for edge in self.edges:
            if edge.prev == nodes[0] and edge.next == nodes[1]:
                return edge
            if edge.prev == nodes[1] and edge.next == nodes[0]:
                return edge
        return 0

    def dijkstra(self, nodes):
        while len(nodes) > 0:
            node = nodes.pop(0)
            for next_node in node.neighbours:
                their_edge = self.search_edge([node, next_node])
                if next_node.min_distance is None or next_node.min_distance > node.min_distance + their_edge.distance:
                    next_node.min_distance = node.min_distance + their_edge.distance
                    next_node.from_node = node
                
                if not next_node.visit:
                    nodes.append(next_node)
            node.visit = True

    def find_node_from(self, nodes):
        max_trans = 0
        result_node = None

        for nn in nodes:
            your_neighs = list(filter(lambda x_node: x_node.transform, nn.neighbours))
            if len(your_neighs) > max_trans:
                max_trans = len(your_neighs)
                result_node = nn
        
        tr = nodes.remove(result_node)
        return result_node, nodes
    
    def find_all_next_nodes(self, from_node):
        result_nodes = []
        for node in self.nodes:
            if node.from_node is not None and node.from_node == from_node and not node.transform:
                result_nodes.append(node)
        
        result_nodes = sorted(result_nodes, key=lambda x_node: x_node.min_distance)
        return result_nodes
    
    def test_transform(self, nodes):
        for node in nodes:
            all_results = []
            rows = []
            a = node.params - node.from_node.params
            norm_of_a = Node.find_norma(a)
            for neigh_node in node.from_node.neighbours:
                if not neigh_node.transform:
                    continue
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
            node.new_params = res.x
            node.transform = True

    def transform_nodes(self, nodes, result, choosen_node):
        return_nodes = []
        while len(nodes) > 0:
            from_node, nodes = self.find_node_from(nodes)
            transform_nodes = self.find_all_next_nodes(from_node)
            self.test_transform(transform_nodes)

            nodes.extend(transform_nodes)
            return_nodes.extend(transform_nodes)

            # drawing

            # list_transform_nodes = np.array([x_node.new_params for x_node in transform_nodes])

            # plt.scatter(result[0, 0], result[0, 1], color="r")
            # plt.scatter(result[1:, 0], result[1:, 1])
            # plt.scatter(from_node.new_params[0], from_node.new_params[1], color="b")
            # plt.scatter(list_transform_nodes[:, 0], list_transform_nodes[:, 1], color="g")
            # plt.show()

            # fir = plt.figure()
            # ax = plt.axes(projection = '3d')

            # neigh_params = np.array([x_node.params for x_node in nodes])
            # list_transform_nodes = np.array([x_node.params for x_node in transform_nodes])

            # ax.scatter(choosen_node.params[0], choosen_node.params[1], choosen_node.params[2], color="r")
            # ax.scatter(neigh_params[:, 0], neigh_params[:, 1], neigh_params[:, 2])
            # ax.scatter(from_node.params[0], from_node.params[1], from_node.params[2], color="b")
            # ax.scatter(list_transform_nodes[:, 0], list_transform_nodes[:, 1], list_transform_nodes[:, 2], color="g")

            # plt.show()

        return return_nodes

    def transform_other_points(self):
        new_nodes = list(filter(lambda x: x.min_distance is not None and not x.transform, self.nodes))
        new_nodes = sorted(new_nodes, key=lambda x: x.min_distance)

        index = 0
        draw_nodes = []
        while len(new_nodes):
            if new_nodes[index].from_node is None or not new_nodes[index].from_node.transform:
                index += 1
                continue
            new_node = new_nodes.pop(index)
            index = 0
            all_results = []
            rows = []
            draw_points = []
            colors = []
            a = new_node.params - new_node.from_node.params
            draw_points.append(new_node.from_node.new_params)
            colors.append('b')
            norm_of_a = Node.find_norma(a)
            for neigh_node in new_node.from_node.neighbours:
                if not neigh_node.transform:
                    continue
                b = neigh_node.params - new_node.from_node.params
                current_cos = np.dot(a, b) / (Node.find_norma(a) * Node.find_norma(b))
                all_results.append(current_cos)
                diff = neigh_node.new_params - new_node.from_node.new_params
                row = diff.T / (norm_of_a * Node.find_norma(diff))
                rows.append(row)

                draw_points.append(neigh_node.new_params)
                colors.append("r")

            x0 = new_node.from_node.new_params
            # minimize(self._fitness_wrapper, x0.reshape(-1), args=(individ, self, shp))
            res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(np.array(rows), np.array(all_results)))

            new_node.new_params = res.x
            new_node.transform = True
            draw_nodes.append(new_node.new_params)
            draw_points.append(new_node.new_params)
            colors.append("g")

            draw_points = np.array(draw_points)

            # plt.scatter(draw_points[:, 0], draw_points[:, 1], color=colors)
            # plt.show()
        
        return np.array(draw_nodes)



    # @njit
    def find_ED(self, eps):
        N = len(self.nodes)
        edgesg = []
        maxval = None
        # print(graph.shape)
        for i in range(N):
            for j in range(i+1, N):
                val = Edge(self.nodes[i], self.nodes[j])
                edgesg.append(val)
                if not maxval:
                    maxval = val.distance
                elif val.distance > maxval:
                    maxval = val.distance

        self.max_edge = maxval

        for i, edge in enumerate(edgesg):
            if edge.distance/maxval <= eps:
                edge.prev.__add_neighbour__(edge.next)
                edge.next.__add_neighbour__(edge.prev)
                self.edges.append(edge)

    def print_info_edges(self):
        for edge in self.edges:
            print(edge.prev.name, edge.next.name)

    def draw(self):
        draw_graph = nx.Graph()
        draw_graph.add_nodes_from([str(j) for j in range(len(self.nodes))])
        edge_xyz = []
        for edge in self.edges:
            draw_graph.add_edge(edge.prev.name, edge.next.name, weight=edge.distance)
            edge_xyz.append([edge.prev.params, edge.next.params])
        
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
    
    def __init__(self, prev_point, next_point):
        self.prev = prev_point
        self.next = next_point
        self.distance = Edge.distance(self.prev.params, self.next.params)
        self.select = False

    def __eq__(self, __value: object) -> bool:
        if self.next == __value.next and self.prev == __value.prev:
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



    

        