from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as colors_tool

import topo as tp
import base.methods as mth
from base.network import Graph

import umap


if __name__ == "__main__":
    R = 5
    n = 1000

    dn = 1

    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)

    colormap = cm.viridis
    colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]

    sorted_data = np.array([x, y, z])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]

    # полусфера
    # new_x = []
    # new_y = []
    # new_z = []

    # for i in range(n):
    #     if z[i] >= 0:
    #         new_x.append(x[i])
    #         new_y.append(y[i])
    #         new_z.append(z[i])

    # x = np.array(new_x)
    # y = np.array(new_y)
    # z = np.array(new_z)

    # свис ролл
    # data = make_swiss_roll(n_samples=n)
    # # data = make_swiss_roll(n_samples=n, hole=True)

    # x = data[0][:, 0]
    # y = data[0][:, 1]
    # z = data[0][:, 2]
    # colors = data[1].copy()

    # циллиндр
    # theta = np.random.random(size=n) * 2 * np.pi
    # # theta = np.linspace(0, 0.2 * np.pi, n) * 2 * np.pi
    # z = np.random.random(size=n)
    # # z = np.linspace(-1, 5, n)
    # x = R * np.cos(theta)
    # y = R * np.sin(theta)

    # min_x = np.min(x)
    # min_y = np.min(y)

    # colormap = cm.viridis
    # colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]
    
    # sorted_data = np.array([x, y, z])
    # # sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[1]/np.sqrt(parameters[0] ** 2 + parameters[1] ** 2)))
    # sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: [parameters[0], parameters[1], parameters[2]]))

    # x = sorted_data[:, 0]
    # y = sorted_data[:, 1]
    # z = sorted_data[:, 2]

    # hepta

    # raw_data = loadarff("data/atom.arff")
    # df_data = pd.DataFrame(raw_data[0])

    # each_color = np.unique(df_data["class"])
    # colormap = cm.viridis
    # all_colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(each_color))]

    # colors = df_data["class"].to_numpy()
    # for i, elem in enumerate(each_color):
    #     colors[colors==elem] = all_colors[i]

    # x = df_data["x"]
    # y = df_data["y"]
    # z = df_data["z"]


    # gauss

    # x = np.random.uniform(-1, 1, size=n)
    # y = np.random.uniform(-1, 1, size=n)
    # theta = np.pi / 2

    # # z = -np.sqrt(x ** 2 + y ** 2) * (1 / np.tan(theta))
    # z = np.exp(-x ** 2 - y ** 2)

    # # for i in range(200):
    # #     z[i] = -z[i]

    # colormap = cm.viridis
    # colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]

    # sorted_data = np.array([x, y, z])
    # sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    # x = sorted_data[:, 0]
    # y = sorted_data[:, 1]
    # z = sorted_data[:, 2]


    data = np.array([x, y, z]).T

    # real data
    dn = 1

    data = load_breast_cancer(return_X_y=True, as_frame=True)
    feature = data[0]
    target = data[1]


    raw_data = loadarff("data/electricity-normalized.arff")
    df_data = pd.DataFrame(raw_data[0])
    df_data['day'] = df_data['day'].astype('int32')
    up_data = df_data[df_data['class']==b'UP'][:2500]
    down_data = df_data[df_data['class']==b'DOWN'][:2500]

    work_data = up_data[:2000]
    work_data = work_data.append(down_data[:2000])
    work_data = work_data.append(up_data[2000:])
    work_data = work_data.append(down_data[2000:])

    # work_data = df_data[:5000]
    # work_data = df_data

    target = work_data['class'].to_numpy()
    target[target==b'UP'] = 1
    target[target==b'DOWN'] = 0
    target = target.astype(dtype=int)
    feature = work_data[['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']]

    # raw_data = loadarff("data/phpSSK7iA.arff")
    # df_data = pd.DataFrame(raw_data[0])
    # target = df_data['target']
    # target[target==b'1'] = 1
    # target[target==b'0'] = 0
    # target = target.astype(int)

    # ks = list(df_data.keys())
    # ks = ks[:-1]
    # feature = df_data[ks]

    data, avg_of_data, var_of_data = mth.prebording_data(feature.values)

    # embedding = umap.UMAP(n_neighbors=500,
    #                   min_dist=0.4,
    #                   metric='cosine').fit_transform(data)
    
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=target)
    # plt.show()
    
    # print("find")


    graph = Graph(data, target, 3)
    print(len(graph.edges))
    # graph.drawing.draw_graph()

    # kernel = tp.tpgraph.Kernel(n_neighbors=10, n_jobs=1, metric='cosine')
    # kernel.fit(data)

    diff_op_isomap = tp.lt.Projector(n_components=2, projection_method='MAP', metric='precomputed').fit_transform(graph.kernel.P)
    tp.pl.scatter(diff_op_isomap, labels=target, pt_size=6)

    print("finish")

    # choose_node = None
    # for node in graph.nodes.values():
    #     if not choose_node:
    #         choose_node = node
    #     elif len(list(graph.neighbors(node["name"]))) > len(list(graph.neighbors(choose_node["name"]))):
    #         choose_node = node

    # graph.check_visible_neigh_with_ts([choose_node])
    # print(len(graph.edges))

    # choosen_nodes = graph.search_nodes(dn, choose_node)

    # base_points = choosen_nodes.copy()

    # for choosen_node in base_points:
    #     choosen_node["min_distance"] = 0
    #     choosen_node["from_node"] = None
    #     graph.dijkstra([choosen_node])

    # print("end")

    # keys = ["r", "b", "g"]
    # picture = {}

    # for index_key, choosen_node in enumerate(base_points):     
    #     data_for_pca, colors_pca = graph.get_data_for_pca(choosen_node)
    #     data_for_pca = np.array(data_for_pca)

    #     # temp_avg = []
    #     # temp_var = []

    #     # for i in range(len(data_for_pca[0])):
    #     #     temp_avg.append(np.average(data_for_pca[:, i]))
    #     #     temp_var.append(np.var(data_for_pca[:, i]))

    #     # graph.avg = temp_avg
    #     # graph.var = temp_var

    #     # start_count_components = data_for_pca.shape[1]

    #     # for i, value in enumerate(data_for_pca):
    #     #     data_for_pca[i] = (value - graph.avg) / graph.var

    #     # pca = PCA(n_components=start_count_components)
    #     # pca.fit(data_for_pca)
    #     # values = pca.singular_values_

    #     # diffs = values[:-1] - values[1:]
    #     # print(diffs)
    #     # mx = np.max(diffs)
    #     # splt_index = np.argmax(diffs)

    #     # newN = len(data_for_pca[:(splt_index+1)])
    #     newN = 2
    #     # print(newN)

    #     pca = PCA(n_components=newN)
    #     pca.fit(data_for_pca)
    #     print(pca.singular_values_)
    #     result = pca.transform(data_for_pca)

    #     # center = result[0]

    #     # for i in range(len(result)-1, -1, -1):
    #     #     result[i] = result[i] - result[0]

    #     graph.set_new_params(choosen_node, result)

    #     graph.find_raw_params(pca)

    #     plt.scatter(result[0, 0], result[0, 1], c=["r"])
    #     plt.scatter(result[1:, 0], result[1:, 1])
    #     plt.show()

    #     nodes = [graph.nodes[index_node] for index_node in graph.neighbors(choosen_node["name"])]
    #     other_nodes = graph.transform_nodes(nodes, result, choosen_node)
    #     print(f"LEN other POINTS: {len(other_nodes)} + {len(result)}")

    #     a = [x_node["params"] for x_node in nodes]
    #     b = [x_node["params"] for x_node in other_nodes]

    #     a.extend(b)

    #     picture[keys[index_key]] = np.array(a)

    #     # other_nodes = graph.get_other_nodes()
    #     # other_nodes = pca.transform(other_nodes)
        
    #     # other_points = np.array(other_nodes)
    #     other_points = np.array([x_node["new_params"] for x_node in other_nodes])
    #     other_colors = np.array([x_node["color"] for x_node in other_nodes])

    #     # plt.scatter(result[0, 0], result[0, 1], color="r")
    #     # plt.scatter(result[1:, 0], result[1:, 1])
    #     plt.scatter(result[:, 0], result[:, 1], c=colors_pca)
    #     try:
    #         # plt.scatter(other_points[:, 0], other_points[:, 1], color="g")
    #         plt.scatter(other_points[:, 0], other_points[:, 1], c=other_colors)
    #     except Exception as e:
    #         print(f"FALL: {e}")
    #         print(other_points)
    #     plt.show()

    #     # info_for_draw = [choose_node]
    #     # info_for_draw.extend(choose_node.neighbours)
    #     # info_for_draw.extend(other_nodes)
    #     # graph.drawing.draw_graph(mode=1, data=info_for_draw)