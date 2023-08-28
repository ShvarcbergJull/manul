from numba import njit
import numpy as np
import matplotlib.pyplot as plt

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
        
        return result



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

    def get_names(self, node):
        names = node.name
        for n_node in self.nodes:
            names.append(n_node.name)

        return names
    
    def check_visible_neigh(self):
        for node in self.nodes:
            for neigh_node in node.neighbours:
                for neigh_node2 in node.neighbours:
                    if neigh_node == neigh_node2:
                        continue
                    
                    value = np.dot(node.params - neigh_node2.params, neigh_node.params - neigh_node2.params)
                    if value < 0:
                        self.delete_edge([node, neigh_node2])
                        node.__delete_neighbour__(neigh_node2)
                        neigh_node2.__delete_neighbour__(node)

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
                    self.from_node = node
                
                if not next_node.visit:
                    nodes.append(next_node)
            node.visit = True
        
        

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
                # edge.prev.count_neighbors += 1
                # edge.next.count_neighbors += 1
                self.edges.append(edge)
            # else:
            #     edge.distance = edge.distance/maxval

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



    

        