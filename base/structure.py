from numba import njit, typed
import numpy as np
from itertools import combinations, product

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import networkx as nx
import topo as tp
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
# from line_profiler import profile

# import lib

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
def checking(node_param, neighbour_params, neighbours):
    neigh_2 = node_param - neighbour_params
    result = np.diag(np.dot(neighbours, neigh_2.T))
    if len(result[result < 0]) > 0:
        return False
    return True

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
        
        self._source_data = data

        self.matrix_connect = np.zeros((self.number_of_nodes(), self.number_of_nodes()))
        self.find_ED(0.4)
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

        maxval = np.max(eds)
        for i in range(N):
            for j in range(i, N):
                self.matrix_connect[i][j] = eds[i][j]
                self.matrix_connect[j][i] = eds[i][j]
                if eds[i][j]/maxval <= eps:
                    self.add_edge(i, j, weight=eds[i][j])

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
    
    # @profile
    def check_visible_neigh_b(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])

            for check_this_index in neighbours:
                check_this = self.nodes[check_this_index[0]]
                if check_this["select"]:
                    continue
                flag = False
                for neighbour_index in self.neighbors(current_node["name"]):
                    neighbour = self.nodes[neighbour_index]
                    if check_this["name"] == neighbour["name"]:
                        continue
                    value = np.dot(current_node["params"] - neighbour["params"], check_this["params"] - neighbour["params"])

                    if value < 0:
                        flag = True
                        break

                if flag:
                    self.remove_edge(current_node["name"], check_this["name"])
                else:
                    start_nodes.append(check_this)
            # new_neighbours_indexs = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])
            # new_neighbours_indexs = filter(lambda index: not self.nodes[index[0]]["select"], new_neighbours_indexs)
            # start_nodes.extend([self.nodes[x[0]] for x in new_neighbours_indexs])

    def check_visible_neigh_gh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours_indexes = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])
            neighbours_indexes = np.array(list(zip(*neighbours_indexes))[0])
            add_params = np.array([self.nodes[node]["params"] for node in neighbours_indexes])
            neighbours = np.array(current_node["params"]) - add_params

            for i, elem in enumerate(neighbours_indexes):
                check_this = self.nodes[elem]
                neigh_2 = np.array(check_this["params"]) - add_params
                result = np.diag(np.dot(neighbours, neigh_2.T))
                if len(result[result < 0]) > 0:
                    self.remove_edge(current_node["name"], check_this["name"])
                else:
                    if not check_this["select"]:
                        start_nodes.append(check_this)

    def check_visible_neigh(self, start_nodes):
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours_indexes = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])
            neighbours_indexes = np.array(list(zip(*neighbours_indexes))[0])
            add_params = np.array([self.nodes[node]["params"] for node in neighbours_indexes])
            neighbours = np.array(current_node["params"]) - add_params

            for i, elem in enumerate(neighbours_indexes):
                check_this = self.nodes[elem]
                if check_this["select"]:
                    continue
                result = checking(check_this["params"], add_params, neighbours)
                if not result:
                    self.remove_edge(current_node["name"], check_this["name"])
                else:
                    start_nodes.append(check_this)

    def check_visible_neigh_with_ts(self, start_nodes):
        pca = PCA(n_components=2)
        while len(start_nodes) > 0:
            current_node = start_nodes.pop(0)
            current_node["select"] = True
            if len(list(self.neighbors(current_node["name"]))) == 0:
                continue
            neighbours = sorted(self[current_node["name"]].items(), key=lambda edge: edge[1]["weight"])

            for check_this_index in neighbours:
                check_this = self.nodes[check_this_index[0]]
                if check_this["select"]:
                    continue
                flag = False
                for neighbour_index in self.neighbors(current_node["name"]):
                    neighbour = self.nodes[neighbour_index]
                    if check_this["name"] == neighbour["name"]:
                        continue
                    new_params = pca.fit_transform([current_node["params"] - neighbour["params"], check_this["params"] - neighbour["params"]])
                    value = np.dot(new_params[0], new_params[1])

                    if value < 0:
                        flag = True
                        break

                if flag:
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