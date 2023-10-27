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


if __name__ == "__main__":
    R = 5
    n = 1000

    dn = 2

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

    data, avg_of_data, var_of_data = mth.prebording_data(feature.values)

    graph = Graph(data, target)
    # graph.drawing.draw_graph()

    # kernel = tp.tpgraph.Kernel(n_neighbors=10, n_jobs=1, metric='cosine')
    # kernel.fit(data)

    diff_op_isomap = tp.lt.Projector(projection_method='Isomap', metric='precomputed').fit_transform(graph.kernel.P)
    tp.pl.scatter(diff_op_isomap, labels=target, pt_size=6)

    print("finish")
    