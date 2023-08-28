import methods
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import networkx as nx

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

from base.structure import Graph

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
    n = 200

    # rs = R*np.sqrt(np.random.random(size=n))
    # thetas = theta = np.random.random(size=n) * 2 * np.pi

    # x = rs * np.cos(thetas)
    # y = rs * np.sin(thetas)
    # z = np.sqrt(R**2 - x**2 - y**2)

    # x = np.concatenate((x, x))
    # y = np.concatenate((y, y))
    # z = np.concatenate((z, -z))

    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(x, y, z)

    plt.show()

    data = np.array([x, y, z]).T
    graph = Graph(data=data)

    print(graph)

    graph.draw()
    print(len(graph.edges))

    # graph.print_info_edges()

    graph.check_visible_neigh()
    
    graph.draw()
    print(len(graph.edges))

    choosen_node = None
    for node in graph.nodes:
        if not choosen_node:
            choosen_node = node
        elif len(node.neighbours) > len(choosen_node.neighbours):
            choosen_node = node

    choosen_node.min_distance = 0

    graph.dijkstra([choosen_node])

    print("end")


    data_for_pca = choosen_node.get_data_for_pca()
    data_for_pca = np.array(data_for_pca)

    start_count_components = data_for_pca.shape[1]

    pca = PCA(n_components=start_count_components)
    pca.fit(data_for_pca)
    values = pca.singular_values_

    diffs = values[:-1] - values[1:]
    print(diffs)
    mx = np.max(diffs)
    splt_index = np.argmax(diffs)

    newN = len(data_for_pca[:(splt_index+1)])
    print(newN)

    pca = PCA(n_components=newN)
    pca.fit(data_for_pca)
    print(pca.singular_values_)
    result = pca.transform(data_for_pca)
    plt.scatter(result[0, 0], result[0, 1], color="r")
    plt.scatter(result[1:, 0], result[1:, 1])
    plt.show()

    

'''
    eps = 0.3
    data = np.array([x, y, z]).T
    graph = methods.find_ED(data, eps)

    graph[graph>eps] = 0


    # print(graph)

    # raw_data = loadarff("data/phpSSK7iA.arff")
    # df_data = pd.DataFrame(raw_data[0])
    # target = df_data['target']
    # target[target==b'1'] = 1
    # target[target==b'0'] = 0
    # target = target.astype(int)

    # ks = list(df_data.keys())
    # ks = ks[:-1]
    # feature = df_data[ks]
    # # end block with reading real data

    # dims = len(feature.keys()) # смотрим кол-во примеров

    # train_features, train_target, test_features, test_target = methods.split_data_TT(feature=feature, target=target) # делим данные на обучение и тест

    # graph = methods.find_ED(train_features.numpy(), 0.3)

    # print(graph.shape)


    # G = nx.Graph()
    # N = len(data)
    # print(N)
    # for i in range(N):
    #     for j in range(i, N):
    #         # print(i, j)
    #         if graph[i][j] != 0:
    #             G.add_edge(i, j, weight=graph[i][j])

    # weights = nx.get_edge_attributes(G,'weight').values()
    # nx.draw(G)
    # plt.show()

    # PCA

    count_neigbors, id_max = methods.count_neighbors(graph)
    # print(count_neigbors, id_max)

    data_for_pca = methods.point_with_neighbors(data=data, id_point=id_max, graph=graph)
    data_for_pca = np.array(data_for_pca)
    # print(data_for_pca)

    print(data.shape)
    print(data_for_pca.shape)

    pca = PCA(n_components=3)
    pca.fit(data_for_pca)
    values = pca.singular_values_

    different = 0

    diffs = values[:-1] - values[1:]
    print(diffs)
    mx = np.max(diffs)
    splt_index = np.argmax(diffs)

    newN = len(data_for_pca[:(splt_index+1)])
    print(newN)

    pca = PCA(n_components=newN)
    pca.fit(data_for_pca)
    print(pca.singular_values_)
    result = pca.transform(data_for_pca)
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()
    # print()


 '''       



'''
    # создание бейслайн модели
    b_model = baseline(dims)
    b_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(b_model.parameters(), lr=1e-4, eps=1e-4)
    b_model.train()

    # создание второй модели для проверки учитывания с 
    x_model = baseline(dims)
    criterion = nn.BCELoss()
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-4, eps=1e-4)
    x_model.train()

    settings = {"lmd": 0, "num_epochs": 500, "batch_size": 750}

    new_model = methods.fit_nn(train_features, train_target, b_model, optimizer, b_criterion, manifold=False, lmd=0)
'''

