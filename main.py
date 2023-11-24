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
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

from base.network import Graph
# from network2 import Graph

def baseline(dim, cl):
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
        # nn.LogSoftmax(dim=1)
        nn.Sigmoid()
    )

    return baseline_model


def find_graph_loss(graph, f_x, indexs):
    # adject = graph.A[indexs][:, indexs]
    # krdf = graph.K[indexs][:, indexs]

    # laplassian = adject - krdf
    laplassian = graph.kernel.L.todense()[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]

def matr_gen_graph(graph, f_x):
    adject = graph.A
    krdf = graph.K

    laplassian = adject - krdf
    # laplassian = graph.kernel.L.todense()[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss


if __name__ == "__main__":
    # рисование сферы

    R = 5
    n = 400
    dims = 3

    dn = 2

    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)

    colormap = cm.viridis
    # colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]
    colors = np.linspace(0, 0.9, n)
    # colors = list(range(n))

    sorted_data = np.array([x, y, z])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]

    colors[colors <= 0.5] = 0
    colors[colors > 0.5] = 1


    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x1 = R * np.cos(theta) * np.sin(phi)
    y2 = R * np.sin(theta) * np.sin(phi)
    z3 = R * np.cos(phi)

    colormap = cm.viridis
    # test_colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]
    test_colors = np.linspace(0, 0.9, n)
    # test_colors = list(range(n))

    sorted_data = np.array([x1, y2, z3])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    x1 = sorted_data[:, 0]
    y2 = sorted_data[:, 1]
    z3 = sorted_data[:, 2]

    test_colors[test_colors <= 0.5] = 0
    test_colors[test_colors > 0.5] = 1

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
    # colors[colors<10] = 0
    # colors[colors>=10] = 1
    # # colors[colors<6] = 4
    # # colors[(colors < 8) * (colors >= 6)] = 5
    # # colors[(colors < 12) * (colors >= 8)] = 10
    # # colors[colors>=12] = 14

    # test_data = make_swiss_roll(n_samples=n)
    # x1 = test_data[0][:, 0]
    # y2 = test_data[0][:, 1]
    # z3 = test_data[0][:, 2]
    # test_colors = test_data[1].copy()
    # test_colors[test_colors<10] = 0
    # test_colors[test_colors>=10] = 1
    # test_colors[test_colors<6] = 4
    # test_colors[(test_colors < 8) * (test_colors >= 6)] = 5
    # test_colors[(test_colors < 12) * (test_colors >= 8)] = 10
    # test_colors[test_colors >=12] = 14


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

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(x, y, z, c=colors)

    plt.show()

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(x1, y2, z3, c=test_colors)

    plt.show()

    data = np.array([x, y, z]).T
    graph = Graph(data=data, colors=colors, n_neighbors=5)

    data = torch.from_numpy(data)
    test_data = torch.from_numpy(np.array([x1, y2, z3]).T)
    train_target = torch.from_numpy(np.array(colors))
    test_target = torch.from_numpy(np.array(test_colors))

    print(graph)

    graph.drawing.draw_graph()
    print(len(graph.edges))

    choose_node = None
    for node in graph.nodes.values():
        if not choose_node:
            choose_node = node
        elif len(list(graph.neighbors(node["name"]))) > len(list(graph.neighbors(choose_node["name"]))):
            choose_node = node

    graph.check_visible_neigh([choose_node])
    
    # graph.drawing.draw()
    # print(len(graph.edges))
    graph.drawing.draw_graph()

    # result_matrix = matr_gen_graph(graph, data)
    F1 = []
    F2 = []

    for k in range(10):
        b_model = baseline(dims, len(np.unique(colors)))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(b_model.parameters(), lr=1e-4, eps=1e-4)
        b_model.train()

        x_model = baseline(dims, len(np.unique(colors)))
        criterion = nn.BCELoss()
        x_optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-4, eps=1e-4)
        x_model.train()
        
        batch_size = 25
        num_epochs = 100 
        min_loss, t = np.inf, 0
        threshold = None
        for epoch in range(num_epochs):
            permutation = torch.randperm(data.size()[0])
            loss_list = []
            for i in range(0, len(train_target), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = data[indices]
                target_y = train_target[indices]
                target_y = target_y.to(torch.float64)
                optimizer.zero_grad()
                output = b_model(batch_x)

                loss = criterion(output, target_y.reshape_as(output))
                fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                if not threshold:
                    threshold = thresholds[ix]
                else:
                    threshold = np.mean([thresholds[ix], threshold])
                
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)

            t += 1
            print('Surface training t={}, loss={}'.format(t, loss_mean))

        b_model.eval()
        nn_out_s = b_model(test_data)
        nn_out = nn_out_s.detach().numpy()
        nn_out = np.where(nn_out > threshold, 1, 0)
        cm = confusion_matrix(test_target.reshape(-1), nn_out.reshape(-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"gik/{k}und0")
        metric = f1_score(test_target.reshape(-1), nn_out.reshape(-1), average=None)
        # print(metric)
        F1.append(list(metric))

        # fir = plt.figure()
        # ax = plt.axes(projection = '3d')

        # ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=nn_out.reshape(-1))

        # plt.show()



        min_loss, t = np.inf, 0
        val = np.min([batch_size, len(data)])
        lmd = 1/(val ** 2)
        for epoch in range(num_epochs):
            permutation = torch.randperm(data.size()[0])
            loss_list = []
            lap_list = []
            for i in range(0, len(train_target), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, target_y = data[indices], train_target[indices]
                target_y = target_y.to(torch.float64)
                x_optimizer.zero_grad()
                output = x_model(batch_x)

                add_loss = find_graph_loss(graph, output.detach().numpy(), indices)
                loss = criterion(output, target_y.reshape_as(output))

                fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                if not threshold:
                    threshold = thresholds[ix]
                else:
                    threshold = np.mean([thresholds[ix], threshold])
                loss += lmd * torch.tensor(add_loss[0, 0])
                
                loss.backward()
                x_optimizer.step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)

            t += 1
            print('Surface training t={}, loss={}'.format(t, loss_mean))

        x_model.eval()
        nn_out_s = b_model(test_data)
        nn_out = nn_out_s.detach().numpy()
        nn_out = np.where(nn_out > threshold, 1, 0)
        cm = confusion_matrix(test_target.reshape(-1), nn_out.reshape(-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"gik/{k}und1")
        metric1 = f1_score(test_target.reshape(-1), nn_out.reshape(-1), average=None)
        F2.append(list(metric1))
        # print(metric)
        # print(metric1)

        # fir = plt.figure()
        # ax = plt.axes(projection = '3d')

        # ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=nn_out.reshape(-1))

        # plt.show()
    

    with open("swiss_res.txt", "w") as fl:
        fl.write(str(F1))
        fl.write("\n")
        fl.write(str(F2))

