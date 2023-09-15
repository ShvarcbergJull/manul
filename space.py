import methods
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from numba import njit

import networkx as nx

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll

from base.structure import Graph, Edge

def baseline(dim):
    baseline_model = nn.Sequential(
        nn.Linear(dim, 256, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(256, 64, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(64, 64, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(64, 256, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(256, 1, dtype=torch.float64),
        nn.Sigmoid()
    )

    return baseline_model

if __name__ == "__main__":
    # рисование сферы

    R = 5
    n = 400

    dn = 1

    samp = np.random.uniform(-1, 1, 2*n).reshape((2, n))
    colors = []

    for i in range(len(samp[0])):
        if samp[0][i] > 0:
            if samp[1][i] > 0:
                colors.append("yellow")
            else:
                colors.append("red")
        else:
            if samp[1][i] > 0:
                colors.append("blue")
            else:
                colors.append("gray")

    plt.scatter(samp[0], samp[1], c=colors)

    # for i in range(len(x)):
    #     x2, y2, _ = proj3d.proj_transform(x[i], y[i], z[i], ax.get_proj())
    #     plt.annotate(str(i), (x2, y2))

    plt.show()

    data = np.array(samp).T
    graph = Graph(data=data, colors=colors)

    print(graph)

    # graph.draw2x()
    print(len(graph.edges))

    # graph.print_info_edges()

    # graph.check_visible_neigh()
    
    # graph.draw2x()
    # print(len(graph.edges))

    choosen_nodes = graph.search_node()

    choose_node = None
    for node in graph.nodes:
        if not choose_node:
            choose_node = node
        elif len(node.neighbours) > len(choose_node.neighbours):
            choose_node = node
    
    graph.check_visible_neigh([choose_node])
    
    graph.draw2x()
    print(len(graph.edges))

    # choosen_node = None
    # for node in graph.nodes:
    #     if not choosen_node:
    #         choosen_node = node
    #     elif len(node.neighbours) > len(choosen_node.neighbours):
    #         choosen_node = node

    # index_node = np.random.randint(0, len(graph.nodes))
    # choosen_node = graph.nodes[index_node]

    # base_points = choosen_nodes[:dn]
    base_points = [choose_node]

    bs_points = np.array([x_node.params for x_node in base_points])
    ng_points = []
    points = np.copy(base_points)
    for choose_node_t in points:
        ng_points_temp = [x_node.params for x_node in choose_node_t.neighbours if x_node not in points]
        ng_points.extend(ng_points_temp)
        base_points.extend(choose_node_t.neighbours)
    ng_points = np.array(ng_points)
    # base_points.extend(choose_node.neighbours)
    # just_points = np.array([x_node.params for x_node in choosen_nodes[dn:]])

    just_points = np.array([x_node.params for x_node in choosen_nodes if x_node not in base_points])

    plt.scatter(bs_points[:, 0], bs_points[:, 1], color="r")
    plt.scatter(ng_points[:, 0], ng_points[:, 1], color="g")
    plt.scatter(just_points[:, 0], just_points[:, 1])

    # print(Edge.distance(bs_points[:, 0], bs_points[:, 1]), Edge.distance(bs_points[:, 0], bs_points[:, 2]), Edge.distance(bs_points[:, 1], bs_points[:, 2]))
    print("before deikstra")
    plt.show()

    # base_points = choosen_nodes[:dn]
    base_points = [choose_node]

    for choosen_node_t in base_points:
        choosen_node_t.min_distance = 0
        choosen_node_t.from_node = None
        graph.dijkstra([choosen_node_t])

    # # graph.clear_after_dkstr()
    # # graph.draw2x()

    # check_dictionary = {}
    # for i in range(n):
    #     check_dictionary[str(i)] = 0

    # for choosen_node in base_points:
    #     new_nodes = [choosen_node]
    #     check_dictionary[choosen_node.name] = -3
    #     while len(new_nodes) > 0:
    #         fr_node = new_nodes.pop(0)
    #         for x_node in graph.nodes:
    #             if x_node.from_node is not None and x_node.from_node == fr_node:
    #                 check_dictionary[x_node.name] += 1
    #                 new_nodes.append(x_node)
    # print(check_dictionary)
    print("end")


    keys = ["r", "b", "g"]
    picture = {}

    for index_key, choosen_node in enumerate(base_points):     
        data_for_pca, colors_pca = choosen_node.get_data_for_pca()
        data_for_pca = np.array(data_for_pca)

        # temp_avg = []
        # temp_var = []

        # for i in range(len(data_for_pca[0])):
        #     temp_avg.append(np.average(data_for_pca[:, i]))
        #     temp_var.append(np.var(data_for_pca[:, i]))

        # graph.avg = temp_avg
        # graph.var = temp_var

        start_count_components = data_for_pca.shape[1]

        for i, value in enumerate(data_for_pca):
            data_for_pca[i] = (value - graph.avg) / graph.var

        pca = PCA(n_components=start_count_components)
        pca.fit(data_for_pca)
        values = pca.singular_values_

        diffs = values[:-1] - values[1:]
        print(diffs)
        mx = np.max(diffs)
        splt_index = np.argmax(diffs)

        newN = len(data_for_pca[:(splt_index+1)])
        newN = 2
        print(newN)

        pca = PCA(n_components=newN)
        pca.fit(data_for_pca)
        print(pca.singular_values_)
        result = pca.transform(data_for_pca)

        # center = result[0]

        # for i in range(len(result)-1, -1, -1):
        #     result[i] = result[i] - result[0]

        choosen_node.set_new_params(result)

        graph.find_raw_params(pca)

        plt.scatter(result[0, 0], result[0, 1], color="r")
        plt.scatter(result[1:, 0], result[1:, 1])
        plt.show()

        nodes = choosen_node.neighbours
        other_nodes = graph.transform_nodes(nodes, result, choosen_node)

        a = [x_node.params for x_node in nodes]
        b = [x_node.params for x_node in other_nodes]

        a.extend(b)

        picture[keys[index_key]] = np.array(a)

        # other_nodes = graph.get_other_nodes()
        # other_nodes = pca.transform(other_nodes)
        
        # other_points = np.array(other_nodes)
        other_points = np.array([x_node.new_params for x_node in other_nodes])
        other_colors = np.array([x_node.color for x_node in other_nodes])

        # plt.scatter(result[0, 0], result[0, 1], color="r")
        # plt.scatter(result[1:, 0], result[1:, 1])
        plt.scatter(result[:, 0], result[:, 1], c=colors_pca)
        try:
            # plt.scatter(other_points[:, 0], other_points[:, 1], color="g")
            plt.scatter(other_points[:, 0], other_points[:, 1], c=other_colors)
        except Exception as e:
            print(f"FALL: {e}")
            print(other_points)
        plt.show()

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    count = 0
    for key in picture:
        count += len(picture[key])
        ax.scatter(picture[key][:, 0], picture[key][:, 1], color=key)

    plt.show()

    print(count, len(graph.nodes))