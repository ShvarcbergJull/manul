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

from base.structure_new import Graph

dn = 1

raw_data = loadarff("data/phpSSK7iA.arff")
df_data = pd.DataFrame(raw_data[0])
random_index = np.random.randint(0, 3750, size=1876)
df_data = df_data.iloc[random_index]
target = df_data['target'].to_numpy()
target[target==b'1'] = 1
target[target==b'0'] = 0
target = target.astype(int)

ks = list(df_data.keys())
ks = ks[:-1]
feature = df_data[ks]

print("FEAT", feature.values.shape)
print("TARGET", target)

graph = Graph(data=feature.values, colors=target)

choose_node = None
for node in graph.nodes:
    if not choose_node:
        choose_node = node
    elif len(node.neighbours) > len(choose_node.neighbours):
        choose_node = node

print("starting check")

graph.check_visible_neigh([choose_node])
choosen_nodes = graph.search_nodes(dn, choose_node)

base_points = choosen_nodes.copy()
for choosen_node in base_points:
    choosen_node.min_distance = 0
    choosen_node.from_node = None
    graph.dijkstra([choosen_node])

keys = ["r", "b", "g"]
picture = {}

for index_key, choosen_node in enumerate(base_points): 
    data_for_pca, colors_pca = choosen_node.get_data_for_pca()
    data_for_pca = np.array(data_for_pca)

    start_count_components = np.min(data_for_pca.shape)

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
    print(f"LEN other POINTS: {len(other_nodes)} + {len(result)}")

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