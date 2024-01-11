import os
import sys
from typing import Union
import torch

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

import numpy as np
from numba import njit
import pandas as pd
import logging
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

from base.entities import DataStructureGraph, PopulationGraph, TakeNN
from base.operators.builder import create_operator_map

from generate_simple_data import create_swiss_roll, create_circle

def handler_of_data(feature, target):
    # dims = len(feature.keys())
    dims = feature.shape[-1]
    print(dims)
    try:
        grid_tensors = [torch.tensor(feature[key].values) for key in feature.keys()]
        grid_tensor = torch.stack(grid_tensors)
        grid_tensor = grid_tensor.view(grid_tensor.shape[0], -1).transpose(0, 1)
    except:
        grid_tensor = torch.from_numpy(feature)
    
    # grid_flattened = grid_tensor.view(grid_tensor.shape[0], -1).transpose(0, 1)
    grid_flattened = grid_tensor.to(grid_tensor.to(torch.float64))
    param = len(target) // 100 * 80
    train_features = grid_flattened[:param]
    train_target = torch.tensor(target)[:param]
    test_features = grid_flattened[param:]
    test_target = torch.tensor(target)[param:]

    return train_features, train_target, test_features, test_target, dims


def find_graph_loss(graph_laplassian, f_x, indexs=None):
    if indexs is None:
        laplassian = graph_laplassian
        
    else:
        laplassian = graph_laplassian[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]

@njit
def find_loss_ind(laplassian, f_x, indexs=None):
    if indexs is not None:
        laplassian = laplassian[indexs][:indexs]
    
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)


def exp_real_data2():
    from scipy.io.arff import loadarff
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

    return feature, target

def exp_real_data3():
    from scipy.io.arff import loadarff
    raw_data = loadarff("data/phpSSK7iA.arff")
    df_data = pd.DataFrame(raw_data[0])
    target = df_data['target']
    target[target==b'1'] = 1
    target[target==b'0'] = 0
    target = target.astype(int)

    ks = list(df_data.keys())
    ks = ks[:-1]
    feature = df_data[ks]

    return feature, target


def main(data: Union[str, np.ndarray]):
    # feature, target = exp_real_data2()
    feature, target = exp_real_data3()
    # feature = data[:, :-1]
    # target = data[:, -1]
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    print(train_feature.shape)

    logging.info("Creating base individ...")
    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), n_neighbors=20, eps=0.25)
    base_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=500)
    logging.info("Creating map with operators and population")

    build_settings = {
        'mutation': {
            'simple': dict(intensive=20, increase_prob=1),
        },
        'crossover': {
            'simple': dict(intensive=1, increase_prob=0.3)
        },
        'population': {
            'size': 10
        },
        'fitness': {
            'test_feature': test_feature,
            'test_target': test_target,
            'add_loss_function': find_graph_loss
        }
    }

    create_operator_map(train_feature, base_individ, base_model.copy(), build_settings)

    population = PopulationGraph(iterations=15)
    population.evolutionary()

    base_model.train()

    result1 = base_model.model_settings['model'](test_feature)
    result1 = result1.detach().numpy()
    result1 = np.where(result1 > base_model.threshold, 1, 0)

    result2 = population.base_model.model_settings['model'](test_feature)
    result2 = result2.detach().numpy()
    result2 = np.where(result2 > population.base_model.threshold, 1, 0)
    cm = confusion_matrix(test_target.reshape(-1), result1.reshape(-1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # plt.savefig(f"images/{k}_nn")
    plt.show()

    cm = confusion_matrix(test_target.reshape(-1), result2.reshape(-1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # plt.savefig(f"images/{k}_nn")
    plt.show()

    metric_nn_1 = f1_score(test_target.reshape(-1), result1.reshape(-1), average=None)
    metric_nn_2 = f1_score(test_target.reshape(-1), result2.reshape(-1), average=None)

    with open("example0.txt", "w") as fl:
        fl.write(str(list(metric_nn_1)))
        fl.write("\n")
        fl.write(str(list(metric_nn_2)))


if __name__ == "__main__":
    # data = create_swiss_roll(1000)
    # data = create_circle(5, 5000)
    # data = "data/electricity-normalized.arff"
    data = "data/phpSSK7iA.arff"
    main(data)


