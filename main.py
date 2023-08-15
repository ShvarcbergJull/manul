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
    raw_data = loadarff("data/phpSSK7iA.arff")
    df_data = pd.DataFrame(raw_data[0])
    target = df_data['target']
    target[target==b'1'] = 1
    target[target==b'0'] = 0
    target = target.astype(int)

    ks = list(df_data.keys())
    ks = ks[:-1]
    feature = df_data[ks]
    # end block with reading real data

    dims = len(feature.keys()) # смотрим кол-во примеров

    train_features, train_target, test_features, test_target = methods.split_data_TT(feature=feature, target=target) # делим данные на обучение и тест

    graph = methods.find_ED(train_features.numpy(), 0.3)

    print(graph.shape)

    G = nx.Graph()
    N = len(feature)
    for i in range(N):
        for j in range(i, N):
            G.add_edge(i, j, weight=graph[i][j])

    # weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G)
    plt.show()

    print(graph)

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

