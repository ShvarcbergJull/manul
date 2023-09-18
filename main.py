import methods
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors_tool
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

    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)

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
    # data = make_swiss_roll(n_samples=n, hole=True)

    # x = data[0][:, 0]
    # y = data[0][:, 1]
    # z = data[0][:, 2]
    # colors = data[1].copy()

    # циллиндр
    theta = np.random.random(size=n) * 2 * np.pi
    # theta = np.linspace(0, 0.2 * np.pi, n) * 2 * np.pi
    z = np.random.random(size=n)
    # z = np.linspace(-1, 5, n)
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    colormap = cm.viridis
    colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]
    
    sorted_data = np.array([x, y, z])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: [parameters[1], parameters[0], parameters[2]]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]


    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(x, y, z, c=colors)

    plt.show()

    data = np.array([x, y, z]).T
    graph = Graph(data=data, colors=colors)

    print(graph)

    graph.draw()
    print(len(graph.edges))

    # graph.print_info_edges()

    # graph.check_visible_neigh()
    
    # graph.draw()
    # print(len(graph.edges))

    av_x = (x.max() + x.min()) / 2
    av_y = (y.max() + y.min()) / 2
    av_z = z.max()

    # choose_node = None
    # mn = None
    # for node in graph.nodes:
    #     num_mn = np.abs(av_x - node.params[0]) + np.abs(av_y - node.params[1]) + np.abs(av_z - node.params[2])
    #     if choose_node is None:
    #         mn = num_mn
    #         choose_node = node
    #         continue
        
    #     if num_mn < mn:
    #         choose_node = node
    #         mn = num_mn

    choose_node = None
    for node in graph.nodes:
        if not choose_node:
            choose_node = node
        elif len(node.neighbours) > len(choose_node.neighbours):
            choose_node = node
    
    graph.check_visible_neigh([choose_node])
    
    graph.draw()
    print(len(graph.edges))

    # choosen_node = None
    # for node in graph.nodes:
    #     if not choosen_node:
    #         choosen_node = node
    #     elif len(node.neighbours) > len(choosen_node.neighbours):
    #         choosen_node = node

    # index_node = np.random.randint(0, len(graph.nodes))
    # choosen_node = graph.nodes[index_node]

    choosen_nodes = graph.search_nodes(dn, choose_node)

    base_points = choosen_nodes.copy()
    # base_points = [choose_node]

    bs_points = np.array([x_node.params for x_node in base_points])
    ng_points = []
    points = np.copy(base_points)
    for choose_node in points:
        ng_points_temp = [x_node.params for x_node in choose_node.neighbours if x_node not in points]
        ng_points.extend(ng_points_temp)
        base_points.extend(choose_node.neighbours)
    ng_points = np.array(ng_points)
    # base_points.extend(choose_node.neighbours)
    # just_points = np.array([x_node.params for x_node in choosen_nodes[dn:]])

    just_points = np.array([x_node.params for x_node in graph.nodes if x_node not in base_points])

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(bs_points[:, 0], bs_points[:, 1], bs_points[:, 2], color="r")
    ax.scatter(ng_points[:, 0], ng_points[:, 1], ng_points[:, 2], color="g")
    ax.scatter(just_points[:, 0], just_points[:, 1], just_points[:, 2])

    print(Edge.distance(bs_points[:, 0], bs_points[:, 1]), Edge.distance(bs_points[:, 0], bs_points[:, 2]), Edge.distance(bs_points[:, 1], bs_points[:, 2]))
    print("before deikstra")
    plt.show()

    base_points = choosen_nodes.copy()
    # base_points = [choose_node]

    for choosen_node in base_points:
        choosen_node.min_distance = 0
        choosen_node.from_node = None
        graph.dijkstra([choosen_node])

    # graph.clear_after_dkstr()
    # graph.draw()

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

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    count = 0
    for key in picture:
        count += len(picture[key])
        ax.scatter(picture[key][:, 0], picture[key][:, 1], picture[key][:, 2], color=key)

    plt.show()

    print(count, len(graph.nodes))


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

