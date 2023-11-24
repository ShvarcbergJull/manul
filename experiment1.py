# Файл с дообучением графа

import base.methods as mth

import torch
from scipy.io.arff import loadarff
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll


from base.network import Graph
# from network2 import Graph


def find_graph_loss(graph, f_x, indexs):
    # adject = graph.A[indexs][:, indexs]
    # krdf = graph.K[indexs][:, indexs]

    # laplassian = adject - krdf
    laplassian = graph.kernel.L.todense()[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]


# R = 5
# n = 400
# dims = 3

# dn = 2
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
# # test_colors[test_colors<6] = 4
# # test_colors[(test_colors < 8) * (test_colors >= 6)] = 5
# # test_colors[(test_colors < 12) * (test_colors >= 8)] = 10
# # test_colors[test_colors >=12] = 14

# train_features = torch.from_numpy(np.array([x, y, z]).T)
# train_target = torch.from_numpy(colors)
# test_features = torch.from_numpy(np.array([x1, y2, z3]).T)
# test_target = torch.from_numpy(test_colors)


# формирование данных

# simple data
# data = load_breast_cancer(return_X_y=True, as_frame=True)
# feature = data[0]
# target = data[1]
# ------------------

# real_data

# raw_data = loadarff("data/electricity-normalized.arff")
# df_data = pd.DataFrame(raw_data[0])
# df_data['day'] = df_data['day'].astype('int32')
# up_data = df_data[df_data['class']==b'UP'][:2500]
# down_data = df_data[df_data['class']==b'DOWN'][:2500]

# work_data = up_data[:2000]
# work_data = work_data.append(down_data[:2000])
# work_data = work_data.append(up_data[2000:])
# work_data = work_data.append(down_data[2000:])

# # work_data = df_data[:5000]
# # work_data = df_data

# target = work_data['class'].to_numpy()
# target[target==b'UP'] = 1
# target[target==b'DOWN'] = 0
# target = target.astype(dtype=int)
# feature = work_data[['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']]


with open("result.txt", "w") as fl:
    fl.write("test")
    fl.close()

# третьи данные
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


dims = len(feature.keys())

grid_tensors = [torch.tensor(feature[key].values) for key in feature.keys()]
grid_tensor = torch.stack(grid_tensors)
grid_flattened = grid_tensor.view(grid_tensor.shape[0], -1).transpose(0, 1)
grid_flattened = grid_flattened.to(grid_flattened.to(torch.float64))
grid_flattened[0]

train_features = grid_flattened
train_target = target

param = len(target) // 100 * 80
train_features = grid_flattened[:param]
train_target = torch.tensor(target)[:param]
test_features = grid_flattened[param:]
test_target = torch.tensor(target)[param:]

graph = Graph(train_features.numpy(), train_target.numpy(), n_neighbors=17)

# обучение моделей

F1 = []
F2 = []
for k in range(10):

    base_model_settings, threshold = mth.take_nn(train_features, train_target, dims, 30, 150)
    base_model = base_model_settings["model"]
    baseline_out = base_model(test_features)
    baseline_out = baseline_out.detach().numpy()
    baseline_out = np.where(baseline_out > threshold, 1, 0)
    # cm = confusion_matrix(test_target.reshape(-1), baseline_out.reshape(-1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.savefig(f"images/{k}_base")
    # plt.show()

    metric = f1_score(test_target.reshape(-1), baseline_out.reshape(-1), average=None)
    # print(baseline_out.shape)
    # print(metric)
    F1.append(list(metric))

    nn_model_settings, threshold = mth.take_nn(train_features, train_target, dims, 30, 150, model_settings=base_model_settings, add_loss_func=find_graph_loss, graph=graph, val=np.min([100, len(train_features)]))
    nn_model = nn_model_settings["model"]
    nn_out = nn_model(test_features)
    nn_out = nn_out.detach().numpy()
    nn_out = np.where(nn_out > threshold, 1, 0)
    # cm = confusion_matrix(test_target.reshape(-1), nn_out.reshape(-1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.savefig(f"images/{k}_nn")
    # plt.show()

    metric_nn = f1_score(test_target.reshape(-1), nn_out.reshape(-1), average=None)
    # print(nn_out.shape)
    # print(metric)
    # print(metric_nn)
    F2.append(list(metric_nn))

with open("experiment1_result.txt", "w") as fl:
    fl.write(str(F1))
    fl.write("\n")
    fl.write(str(F2))