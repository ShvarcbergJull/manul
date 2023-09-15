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

    def __delete_neighbour__(self, neigh_node) -> None:
        self.neighbours.remove(neigh_node)

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



class Graph:
    '''
    Class of graph with points

    Arguments:
        data

    Methods:
        ....
    '''

    def __init__(self, data, colors=None) -> None:
        '''
        data must have size (n, m), where n - count experiments and m - count features of point
        '''

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
        self.matrix_connect = np.zeros((len(points), len(points)))
        self.find_ED(0.15)

    @staticmethod
    def _fitness_wrapper(params, *args):
        A, all_cos = args
        parametr = np.dot(A, params) - all_cos.T
        parameter = Node.find_norma(parametr)

        return  parameter ** 2
    
    def average_distance(self, node):
        all_dist = 0
        for nn in node.neighbours:
            edge = self.search_edge([node, nn])
            if edge:
                all_dist += edge.distance
        # return len(node.neighbours)
        # TODO: handle 'division by zero'
        if len(node.neighbours) == 0:
            return 0
        return (all_dist / len(node.neighbours)) / len(node.neighbours)

    def search_node(self):
        ch_node = None
        for node in self.nodes:
            node.number = self.average_distance(node)
        
        ch_node = sorted(self.nodes, key=lambda x_node: x_node.number, reverse=True)
        return ch_node
    
    def search_nodes(self, n, choose_node):
        if n == 1:
            return [choose_node]
        distance_to_other = self.matrix_connect[int(choose_node.name)]
        index_node = np.argmax(distance_to_other)

        new_node = self.nodes[index_node]
        return [choose_node, new_node]
        

    def get_names(self, node):
        names = node.name
        for n_node in self.nodes:
            names.append(n_node.name)

        return names
    
    @staticmethod
    @njit
    def is_reachable(connect_matrix, n, check_i, check_j):
        result_matrix = connect_matrix.copy()

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    result_matrix[i][j] = result_matrix[i][j] or (result_matrix[i][k] and result_matrix[k][j])
        
        if result_matrix[check_i][check_j] == 1:
            return True
        return False
    
    def check_visible_neigh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node.select = True
            if len(current_node.neighbours) == 0:
                continue
            neighbours = []
            for neighbour in current_node.neighbours:
                if neighbour.select:
                    continue
                edge = self.search_edge([current_node, neighbour])
                if edge is None:
                    current_node.__delete_neighbour__(neighbour)
                else:
                    neighbour.dist = edge.distance
                    neighbours.append(neighbour)
            neighbours = sorted(neighbours, key=lambda x: x.dist, reverse=True)

            for check_this in neighbours:
                flag = False
                for neighbour in current_node.neighbours:
                    if check_this == neighbour:
                        continue
                    value = np.dot(current_node.params - neighbour.params, check_this.params - neighbour.params)

                    if value < 0:
                        flag = True
                        break

                if flag:
                    self.delete_edge([current_node, check_this])
                    current_node.__delete_neighbour__(check_this)
                    check_this.__delete_neighbour__(current_node)      
            new_neighbours = filter(lambda x: not x.select, current_node.neighbours)
            start_nodes.extend(sorted(new_neighbours, key=lambda x: x.dist))


            

    def check_visible_neighb(self):
        for node in self.nodes:
            del_list = []
            for neigh_node in node.neighbours:
                for neigh_node2 in node.neighbours:
                    if neigh_node == neigh_node2:
                        continue

                    if neigh_node not in node.neighbours or neigh_node2 not in node.neighbours:
                        continue
                    
                    value = np.dot(node.params - neigh_node.params, neigh_node2.params - neigh_node.params)
                    if value < 0:
                        self.delete_edge([node, neigh_node2])
                        node.__delete_neighbour__(neigh_node2)
                        neigh_node2.__delete_neighbour__(node)

    def clear_after_dkstr(self):
        for node in self.nodes:
            for n_node in node.neighbours:
                if node.from_node is not None and node.from_node == n_node:
                    continue
                if n_node.from_node is not None and n_node.from_node == node:
                    continue
                self.delete_edge([node, n_node])
                node.__delete_neighbour__(n_node)
                n_node.__delete_neighbour__(node)

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
    
    def add_edge(self, nodes):
        edge = self.search_edge(nodes)

        if edge is None:
            edge = Edge(nodes[0], nodes[1])
            self.edges.append(edge)
    
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
                # if next_node.min_distance is not None and next_node.min_distance == 0:
                #     continue
                their_edge = self.search_edge([node, next_node])
                if next_node.min_distance is None or next_node.min_distance > node.min_distance + their_edge.distance:
                    next_node.min_distance = node.min_distance + their_edge.distance
                    next_node.from_node = node
                
                if not next_node.visit:
                    nodes.append(next_node)
            node.visit = True
        
        for node in self.nodes:
            node.visit = False

    def find_node_from(self, nodes):
        max_trans = 0
        result_node = None

        for nn in nodes:
            your_neighs = list(filter(lambda x_node: x_node.transform, nn.neighbours))
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
    
    def test_transform(self, nodes):
        for node in nodes:
            all_results = []
            rows = []
            # a = node.params - node.from_node.params
            a = node.from_node.params - node.params
            norm_of_a = Node.find_norma(a)
            for neigh_node in node.from_node.neighbours:
                if not neigh_node.transform:
                    continue
                # b = neigh_node.params - node.from_node.params
                b = node.from_node.params - neigh_node.params
                current_cos = np.dot(a, b) / (Node.find_norma(a) * Node.find_norma(b))
                all_results.append(90 - current_cos)
                # diff = neigh_node.new_params - node.from_node.new_params
                diff = node.from_node.new_params - neigh_node.new_params
                row = diff.T / (norm_of_a * Node.find_norma(diff))
                rows.append(row)
            x0 = node.from_node.new_params
            cons = ({'type': 'eq',
                'fun' : lambda x: Node.find_norma(x - node.from_node.new_params) - norm_of_a})
            res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(np.array(rows), np.array(all_results)), method='SLSQP', constraints=cons)
            node.new_params = res.x
            node.transform = True

    def find_raw_params(self, pca, center=None):
        for node in self.nodes:
            params = (node.params - self.avg) / self.var
            res = pca.transform([params])
            node.raw_params = res[0]

    def transform_part(self, nodes):
        for node in nodes:
            all_results = []
            rows = []
            # a = node.raw_params - node.from_node.raw_params
            a = node.params - node.from_node.params
            norm_of_a = Node.find_norma(node.raw_params - node.from_node.raw_params)
            # norm_of_a = Node.find_norma(a)
            # transform_nodes = np.array([x_node for x_node in self.nodes if x_node.transform])
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


    def transform_nodes(self, nodes, result, choosen_node):
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

            # drawing

            # list_transform_nodes = np.array([x_node.new_params for x_node in self.nodes if x_node.transform])
            # current_transform_nodes = np.array([x_node.new_params for x_node in transform_nodes])

            # plt.scatter(list_transform_nodes[:, 0], list_transform_nodes[:, 1])
            # plt.scatter(result[0, 0], result[0, 1], color="r")
            # plt.scatter(from_node.new_params[0], from_node.new_params[1], color="b")
            # plt.scatter(current_transform_nodes[:, 0], current_transform_nodes[:, 1], color="g")
            # plt.show()

            # fir = plt.figure()
            # ax = plt.axes(projection = '3d')

            # all_node = [from_node]
            # current_transform_nodes = np.array([x_node.params for x_node in transform_nodes])
            # all_node.extend(transform_nodes)
            # print(all_node)
            # list_transform_nodes = np.array([x_node.params for x_node in self.nodes if x_node.transform and x_node not in all_node])
            # all_node.extend(np.array([x_node for x_node in self.nodes if x_node.transform and x_node not in all_node]))
            # all_nodes = np.array([x_node.params for x_node in self.nodes if x_node not in all_node])

            # ax.scatter(all_nodes[:, 0], all_nodes[:, 1], all_nodes[:, 2], color="gray")
            # ax.scatter(list_transform_nodes[:, 0], list_transform_nodes[:, 1], list_transform_nodes[:, 2])
            # ax.scatter(choosen_node.params[0], choosen_node.params[1], choosen_node.params[2], color="r")
            # ax.scatter(from_node.params[0], from_node.params[1], from_node.params[2], color="b")
            # ax.scatter(current_transform_nodes[:, 0], current_transform_nodes[:, 1], current_transform_nodes[:, 2], color="g")
            
            # plt.show()

        return return_nodes
    
    def get_other_nodes(self):
        result = []
        for node in self.nodes:
            if node.transform:
                continue
            result.append(node.params)
            node.transform = True
        
        return result

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
            self.matrix_connect[int(edge.prev.name)][int(edge.next.name)] = edge.distance/maxval
            self.matrix_connect[int(edge.next.name)][int(edge.prev.name)] = edge.distance/maxval
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

    def draw2x(self):
        draw_graph = nx.Graph()
        draw_graph.add_nodes_from([str(j) for j in range(len(self.nodes))])
        edge_xyz = []
        for edge in self.edges:
            draw_graph.add_edge(edge.prev.name, edge.next.name, weight=edge.distance)
            edge_xyz.append([edge.prev.params, edge.next.params])
        
        edge_xyz = np.array(edge_xyz)

        # 3d spring layout
        pos = nx.spring_layout(draw_graph, dim=2, seed=779)
        # Extract node and edge positions from the layout
        # node_xyz = np.array([pos[v] for v in sorted(draw_graph)])
        node_xy = []
        for node in self.nodes:
            node_xy.append(node.params)
        node_xy = np.array(node_xy)
        # edge_xyz = np.array([(pos[u], pos[v]) for u, v in draw_graph.edges()])

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xy.T, s=100, ec="w")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")


        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")


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



    

        