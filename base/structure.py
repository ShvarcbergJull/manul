from numba import njit, types
import numpy as np
import torch
from itertools import combinations, product

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from progress.bar import Bar
import scipy as sp
# from line_profiler import profile

# import lib

EPS = -1e-9
device = torch.device('cuda')

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

@njit
def find_norma(params):
    result = 0
    for param in params:
        result += np.power(param, 2)
    
    return np.sqrt(result)

@njit
def helper(eds, norm_eds, i, node_neigh, all_nodes, eps, source_data, check_nodes):
    return_values = np.array([])
    indx_neighbours = np.argsort(eds[i])
    # indx_neighbours = np.delete(indx_neighbours, np.argwhere(indx_neighbours == i))
    print(np.argwhere(indx_neighbours != i))
    print(np.argwhere(indx_neighbours != i)[0])
    indx_neighbours = indx_neighbours[np.argwhere(indx_neighbours != i)[0]]
    if len(node_neigh) == 0:
        node_neigh = np.append(node_neigh, indx_neighbours[0])
        all_nodes = np.append(all_nodes, indx_neighbours[0])
    # indx_neighbours  = np.delete(indx_neighbours, np.argwhere(np.intersect1d(indx_neighbours, node_neigh)))
    # indx_neighbours  = np.delete(indx_neighbours, np.argwhere(np.intersect1d(indx_neighbours, check_nodes)))
    indx_neighbours  = indx_neighbours[np.argwhere(np.logical_not(np.intersect1d(indx_neighbours, node_neigh)))[0]]
    indx_neighbours  = indx_neighbours[np.argwhere(np.logical_not(np.intersect1d(indx_neighbours, check_nodes)))[0]]
    for j in indx_neighbours:
        if norm_eds[i][j] <= eps:
            neighbours = source_data[i] - source_data[node_neigh]
            check = source_data[j] - source_data[node_neigh]
            result = np.diag(np.dot(neighbours, check.T))
            if len(result[result < 0]) > 0:
                continue
            node_neigh = np.append(node_neigh, j)
            all_nodes = np.append(all_nodes, j)
            return_values = np.append(return_values, j)
        else:
            break
    
    return (all_nodes, return_values) 
@njit
def checking(node_param, neighbour_params, neighbours):
    neigh_2 = node_param - neighbour_params
    result = np.diag(np.dot(neighbours, neigh_2.T))
    # print(len(result[result < 0]))
    if len(result[result < 0]) > 0:
        return False
    return True

def checking_1(node_param, neighbour_params, neighbours):
    neigh_2 = torch.sub(node_param, neighbour_params)
    neigh_2.to(device)
    result = torch.round(torch.diag(torch.mm(neighbours.type_as(neigh_2), neigh_2.T)), decimals=4)
    # print(len(result[result < 0]))
    if len(result[result < 0]) > 0:
        return False
    return True

@njit
def tets(eds, A, i):
    indx_neighbours = np.argsort(eds[i])
    indx_neighbours = indx_neighbours[np.argwhere(indx_neighbours != i)[:, 0]]
    node_neigh = np.argwhere(A[i] == 1)
    if indx_neighbours[0] not in node_neigh:
        node_neigh = np.append(node_neigh, indx_neighbours[0])
    # indx_neighbours = indx_neighbours[np.argwhere(indx_neighbours != np.intersect1d(indx_neighbours, node_neigh))[:, 0]]
    indx_neighbours = np.array([val for val in indx_neighbours if val not in node_neigh])

    return indx_neighbours, node_neigh

@njit
def create_matrix(half_matr, n, eps):
    result_matrix = np.zeros((n, n))
    k = 0
    maxval = np.max(half_matr)
    
    for i in range(n):
        for j in range(i+1, n):
            if half_matr[k]/maxval > eps:
                result_matrix[i][j] = half_matr[k]
                result_matrix[j][i] = half_matr[k]
            k += 1
    
    return result_matrix


class Graph(nx.Graph):

    def __init__(self, data, colors=None, n_neighbors = 10, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.avg = []
        self.var = []
        i = 0
    
        for i, elem in enumerate(data):
            color = None
            if colors is not None:
                color = colors[i]                
            self.add_nodes_from([(i, {"name": i, "params": elem, "color": color, "select": False})])
        
        self._source_data = sp.sparse.coo_array(data)

        self.matrix_connect = sp.sparse.coo_array(np.zeros((self.number_of_nodes(), self.number_of_nodes())))
        self.A = sp.sparse.coo_array(np.zeros((self.number_of_nodes(), self.number_of_nodes())))
        self._K = sp.sparse.coo_array(np.zeros((self.number_of_nodes(), self.number_of_nodes())))
        print("Start searching ED")
        self.find_ED(0.4)
        # self.find_ED_new(0.4)
        print("finidshed")
        # self.find_ED_other_realisation(0.7)
        self.drawing = Draw(self)

    @staticmethod
    def _fitness_wrapper(params, *args):
        A, all_cos = args
        parametr = np.dot(A, params) - all_cos.T
        parameter = find_norma(parametr)

        return  parameter ** 2
        
    def find_ED(self, eps):
        N = self.number_of_nodes()
        eds = euclidean_distances(self._source_data, self._source_data)
        self.matrix_connect = eds

        maxval = np.max(eds)
        bar = Bar('Search Edges', max=N)
        bar.start()
        for i in range(N):
            indx_neighbours = np.argsort(eds[i])
            indx_neighbours = np.delete(indx_neighbours, np.where(indx_neighbours == i))
            node_neigh = [elem[0] for elem in self[i].items()]
            if indx_neighbours[0] not in node_neigh:
                node_neigh.append(indx_neighbours[0])
            indx_neighbours  = np.delete(indx_neighbours, np.where(np.intersect1d(indx_neighbours, node_neigh)))
            for j in indx_neighbours:
                if eds[i][j]/maxval <= eps:
                    neighbours = self._source_data[i] - self._source_data[node_neigh]
                    check = self._source_data[j] - self._source_data[node_neigh]
                    result = np.diag(np.dot(neighbours, check.T))
                    if len(result[result < 0]) > 0:
                        continue
                    self.add_edge(i, j, weight=eds[i][j])
                    node_neigh.append(j)
                    self.A[i][j] = 1
                    self.A[j][i] = 1
                    self._K[i][j] = eds[i][j]/maxval
                    self._K[j][i] = eds[i][j]/maxval
            bar.next()
        bar.finish()

    def find_ED_new(self, eps):
        N = self.number_of_nodes()
        eds = euclidean_distances(self._source_data, self._source_data)
        self.matrix_connect = eds

        # tets(eds, eps, self._source_data)
        N = len(eds[0])
        maxval = np.max(eds)
        self.A = np.zeros((N, N))
        self._K = np.zeros((N, N))
        bar = Bar('Search Edges', max=N)
        bar.start()
        for i in range(N):
            indx_neighbours, node_neigh = tets(eds, self.A, i)
            for j in indx_neighbours:
                if eds[i][j]/maxval <= eps:
                    neighbours = self._source_data[i] - self._source_data[node_neigh]
                    check = self._source_data[j] - self._source_data[node_neigh]
                    result = np.diag(np.dot(neighbours, check.T))
                    if len(result[result < 0]) > 0:
                        continue
                    self.add_edge(i, j, weight=eds[i][j])
                    node_neigh = np.append(node_neigh, j)
                    self.A[i][j] = 1
                    self.A[j][i] = 1
                    self._K[i][j] = eds[i][j]/maxval
                    self._K[j][i] = eds[i][j]/maxval
            bar.next()
        bar.finish()

            

        # maxval = np.max(eds)
        # bar = Bar('Search Edges', max=N)
        # bar.start()
        # for i in range(N):
        #     indx_neighbours = np.argsort(eds[i])
        #     indx_neighbours = np.delete(indx_neighbours, np.where(indx_neighbours == i))
        #     node_neigh = [elem[0] for elem in self[i].items()]
        #     if indx_neighbours[0] not in node_neigh:
        #         node_neigh.append(indx_neighbours[0])
        #     indx_neighbours  = np.delete(indx_neighbours, np.where(np.intersect1d(indx_neighbours, node_neigh)))
        #     for j in indx_neighbours:
        #         if eds[i][j]/maxval <= eps:
        #             neighbours = self._source_data[i] - self._source_data[node_neigh]
        #             check = self._source_data[j] - self._source_data[node_neigh]
        #             result = np.diag(np.dot(neighbours, check.T))
        #             if len(result[result < 0]) > 0:
        #                 continue
        #             self.add_edge(i, j, weight=eds[i][j])
        #             node_neigh.append(j)
        #             self.A[i][j] = 1
        #             self.A[j][i] = 1
        #             self._K[i][j] = eds[i][j]/maxval
        #             self._K[j][i] = eds[i][j]/maxval
        #     bar.start()
        # bar.finish()

    
    def find_ED_other_realisation(self, eps):
        N = self.number_of_nodes()
        eds = euclidean_distances(self._source_data, self._source_data)

        maxval = np.max(eds)
        norm_eds = eds / maxval
        first = np.argmax([len(norm_eds[i][norm_eds[i] <= eps]) for i in range(N)])
        all_nodes = np.array([first])
        check_nodes = []
        while len(all_nodes) > 0:
            i = all_nodes[0]
            check_nodes.append(i)
            all_nodes = all_nodes[1:]
            node_neigh = np.array([elem[0] for elem in self[i].items()], dtype=np.int64)

            print(i)
            # all_nodes, neighbours = helper(eds, norm_eds, i, node_neigh, all_nodes, eps, self._source_data, check_nodes)
            # print(i)

            # for k in neighbours:
            #     self.add_edge(i, k, weight=eds[i][k])
            
            indx_neighbours = np.argsort(eds[i])
            indx_neighbours = np.delete(indx_neighbours, np.where(indx_neighbours == i))
            print(node_neigh.dtype)
            if len(node_neigh) == 0:
                node_neigh = np.append(node_neigh, indx_neighbours[0])
                all_nodes = np.append(all_nodes, indx_neighbours[0])
            indx_neighbours  = np.delete(indx_neighbours, np.where(np.intersect1d(indx_neighbours, node_neigh)))
            indx_neighbours  = np.delete(indx_neighbours, np.where(np.intersect1d(indx_neighbours, check_nodes)))
            for j in indx_neighbours:
                self.matrix_connect[i][j] = eds[i][j]
                self.matrix_connect[j][i] = eds[i][j]
                if norm_eds[i][j] <= eps:
                    neighbours = self._source_data[i] - self._source_data[node_neigh]
                    check = self._source_data[j] - self._source_data[node_neigh]
                    result = np.diag(np.dot(neighbours, check.T))
                    if len(result[result < 0]) > 0:
                        continue
                    self.add_edge(i, j, weight=eds[i][j])
                    node_neigh = np.append(node_neigh, j)
                    all_nodes = np.append(all_nodes, j)


    def find_graph(self, eps):
        max_value = np.max(self.kernel.SP)
        N = self.number_of_nodes()

        for i in range(N):
            for j in range(i+1, N):
                # if self.kernel.SP[i][j]  / max_value <= eps:
                if self.connected[i, j] == 1 and self.nodes[i]["name"] != self.nodes[j]["name"]:
                    self.add_edge(int(self.nodes[i]["name"]), int(self.nodes[j]["name"]), weight=self.kernel.SP[i][j])

    def search_nodes(self, n, choose_node):
        if n == 1:
            return [choose_node]
        distance_to_other = self.matrix_connect[choose_node["name"]]
        index_node = np.argmax(distance_to_other)

        new_node = self.nodes[index_node]
        return [choose_node, new_node]
    
    def dijkstra(self, nodes):
        while len(nodes) > 0:
            node = nodes.pop(0)
            for next_node_index in self.neighbors(node["name"]):
                next_node = self.nodes[next_node_index]
                # if next_node.min_distance is not None and next_node.min_distance == 0:
                #     continue
                their_edge = self.search_edge([node, next_node])
                if next_node.get("min_distance", None) is None or next_node["min_distance"] > node["min_distance"] + their_edge[2]:
                    self.nodes[next_node_index]["min_distance"] = node["min_distance"] + their_edge[2]
                    self.nodes[next_node_index]["from_node"] = node
                
                if not next_node.get("visit", None):
                    nodes.append(next_node)
            self.nodes[node["name"]]["visit"] = True
        
        for node in self.nodes:
            self.nodes[node]["visit"] = False
    
    def check_visible_neigh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours_indexes = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])
            neighbours_indexes = np.array(list(zip(*neighbours_indexes))[0])
            add_params = torch.from_numpy(np.array([self.nodes[node]["params"] for node in neighbours_indexes]))
            neighbours = torch.sub(torch.Tensor(current_node["params"]), add_params)

            add_params.to(device)
            neighbours.to(device)

            add_params1 = np.array([self.nodes[node]["params"] for node in neighbours_indexes])
            neighbours1 = current_node["params"] - add_params1

            for i, elem in enumerate(neighbours_indexes):
                check_this = self.nodes[elem]
                if check_this["select"]:
                    continue
                prms_check = torch.from_numpy(check_this["params"])
                prms_check.to(device)
                result = checking_1(prms_check, add_params, neighbours)
                # result1 = checking(check_this["params"], add_params1, neighbours1)
                # if result1 != result:
                #     print("opopo")
                if not result:
                    self.remove_edge(current_node["name"], check_this["name"])
                else:
                    start_nodes.append(check_this)

    @staticmethod
    @njit
    def fast_checking(source_data, index_neighbours, index_current_node, index_check_this):
        current_params = source_data[index_current_node]
        check_params = source_data[index_check_this]
        for i_neighbour in index_neighbours:
            if i_neighbour == index_check_this:
                continue
            neigh_params = source_data[i_neighbour]
            value = np.dot(current_params - neigh_params, check_params - neigh_params)

            if value < 0:
                return True
        return False

    def search_edge(self, nodes):
        try:
            distance = self[nodes[0]["name"]][nodes[1]["name"]]["weight"]
        except Exception as e:
            return None
        nodes.append(distance)
        return nodes
    
    def get_data_for_pca(self, from_choose_node):
        result = [from_choose_node["params"]]
        colors = [from_choose_node["color"]]
        for neigh in self.neighbors(from_choose_node["name"]):
            result.append(self.nodes[neigh]["params"])
            colors.append(self.nodes[neigh]["color"])
            self.nodes[neigh]["transform"] = True
        from_choose_node["transform"] = True
        
        return result, colors
    
    def set_new_params(self, from_choosen_node, pca_params):
        from_choosen_node["new_params"] = pca_params[0]
        index_neighbors = list(self.neighbors(from_choosen_node["name"]))
        for i, params in enumerate(pca_params):
            if i == 0:
                continue
            self.nodes[index_neighbors[i - 1]]["new_params"] = params

    def find_raw_params(self, pca, center=None):
        for node_index in self.nodes:
            node = self.nodes[node_index]
            params = (node["params"] - self.avg) / self.var
            res = pca.transform([params])
            self.nodes[node_index]["raw_params"] = res[0]
    
    def find_node_from(self, nodes):
        max_trans = 0
        result_node = None

        for nn in nodes:
            your_neighs = list(filter(lambda x_node: self.nodes[x_node].get("transform", None), self.neighbors(nn["name"])))
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
        for node_index in self.nodes:
            node = self.nodes[node_index]
            if node.get("from_node", None) is not None and node["from_node"] == from_node and not node.get("transform", None):
                result_nodes.append(node)
        
        result_nodes = sorted(result_nodes, key=lambda x_node: x_node["min_distance"])
        return result_nodes

    def transform_part(self, nodes):
        for node in nodes:
            all_results = []
            rows = []
            # a = node.raw_params - node.from_node.raw_params
            a = node["params"] - node["from_node"]["params"]
            norm_of_a = find_norma(node["raw_params"] - node["from_node"]["raw_params"])
            # norm_of_a = Node.find_norma(a)
            # transform_nodes = np.array([x_node for x_node in self.nodes if x_node.transform])
            for neigh_node_index in self.neighbors(node["from_node"]["name"]):
                if not self.nodes[neigh_node_index].get("transform", None):
                    continue
                # b = neigh_node.raw_params - node.from_node.raw_params
                b = self.nodes[neigh_node_index]["params"] - node["from_node"]["params"]
                current_cos = np.dot(a, b) / (find_norma(a) * find_norma(b))
                all_results.append(current_cos)
                diff = self.nodes[neigh_node_index]["new_params"] - node["from_node"]["new_params"]
                row = diff.T / (norm_of_a * find_norma(diff))
                rows.append(row)
            x0 = node["from_node"]["new_params"]
            cons = ({'type': 'eq',
                'fun' : lambda x: find_norma(x) - norm_of_a})
            res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(np.array(rows), np.array(all_results)), method='SLSQP', constraints=cons)
            self.nodes[node["name"]]["new_params"] = res.x + node["from_node"]["new_params"]
            self.nodes[node["name"]]["transform"] = True


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
        
        return return_nodes

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
        colors = np.array([self.graph.nodes[node]["color"] for node in self.graph.nodes])
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