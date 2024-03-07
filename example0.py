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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_squared_error
import matplotlib.pyplot as plt

from base.entities import DataStructureGraph, PopulationGraph, TakeNN, forming_dict
from base.operators.builder import create_operator_map
from base.operators.base import ProgramRun

from generate_simple_data import create_swiss_roll, create_circle
from data_forming import airfoil_exmpl, mammonth_example

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
    param = int(len(target) / 100 * 80)
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


def run_experiment(base_model, test_feature, test_target, number):
    runner = ProgramRun()

    base_model.train()
    result1 = base_model.model_settings['model'](test_feature)
    result1 = result1.detach().numpy()
    result1 = np.where(result1 > base_model.threshold, 1, 0)

    metric_nn_1 = f1_score(test_target.reshape(-1), result1.reshape(-1), average=None)

    population = PopulationGraph(iterations=15)
    population.evolutionary()

    # population.base_model.train(find_graph_loss, population.laplassian)

    result2 = population.base_model.model_settings['model'](test_feature)
    result2 = result2.detach().numpy()
    result2 = np.where(result2 > population.base_model.threshold, 1, 0)

    metric_nn_2 = f1_score(test_target.reshape(-1), result2.reshape(-1), average=None)

    runner.save_confusion_matrix(f"conf_just_model_{number}", data=[test_target, result1])
    runner.save_confusion_matrix(f"conf_EA_model_{number}", data=[test_target, result2])
    # runner.save_confusion_matrix(f"conf_matrix_{number}", data=[test_target, result1], data2=[test_target, result2])
    runner.save_plot(f"fitness_{number}", population.change_fitness)
    runner.save_model(f"model_{number}", population.base_model.model_settings['model'])

    return_dictionary = {
        "f1_score": [list(metric_nn_1), list(metric_nn_2)],
    }

    return return_dictionary

def run_experiment_regression(base_model, test_feature, test_target, number):
    runner = ProgramRun()

    base_model.train()
    result1 = base_model.model_settings['model'](test_feature)
    result1 = result1.detach().numpy()
    runner.save_end_graph(data=result1, name=f"raw_result1_{number}.txt")
    result1 = result1.round().astype("int64")
    runner.save_end_graph(data=result1, name=f"result1_{number}.txt")

    metric_nn_1 = mean_squared_error(test_target.reshape(-1), result1.reshape(-1))

    population = PopulationGraph(iterations=10)
    population.evolutionary(num=number)

    # population.base_model.train(find_graph_loss, population.laplassian)

    result2 = population.base_model.model_settings['model'](test_feature)
    result2 = result2.detach().numpy()
    runner.save_end_graph(data=result2, name=f"raw_result2_{number}.txt")
    result2 = result2.round().astype("int64")
    runner.save_end_graph(data=result2, name=f"result2_{number}.txt")

    metric_nn_2 = mean_squared_error(test_target.reshape(-1), result2.reshape(-1))

    # runner.save_plots(name=f"result_{number}", data=[test_target.reshape(-1), result1.reshape(-1), result2.reshape(-1)], labels=["target", "base", "man"])
    # runner.save_plots(name=f"dif_result_{number}", data=[abs(test_target.reshape(-1) - result1.reshape(-1)), abs(test_target.reshape(-1) - result2.reshape(-1))], labels=["dif_base", "dif_man"])
    runner.save_plot(f"fitness_{number}", population.change_fitness)
    runner.save_model(f"model_{number}", population.base_model.model_settings['model'])

    return_dictionary = {
        "f1_score": [metric_nn_1, metric_nn_2],
    }

    return return_dictionary

def forming_connect(graph):
    # res = {}
    res = []
    for i in graph:
        res.append({'index': i, 'neighbours': [], 'stamp': False})
        # res[i] = {}
    for i in graph:
        for k in graph[i]:
            # res[i][eds[i][k]] = k
            res[k]['neighbours'].append(i)
            res[i]['neighbours'].append(k)

    return res

def searсh_basis(graph, source_data):
    basis = []
    graph = forming_connect(graph)
    temp_graph = list(filter(lambda elem: not elem['stamp'], graph))
    while len(temp_graph) > 0:
        max_index = np.argmax([len(elem['neighbours']) for elem in temp_graph])
        use_index = temp_graph[max_index]['neighbours']
        use_index.append(temp_graph[max_index]['index'])
        average_values = np.average(source_data[use_index], axis=0)
        choose_point = use_index[0]
        for indx in use_index:
            if ((source_data[indx] - average_values) ** 2).sum().sqrt() < ((source_data[choose_point] - average_values) ** 2).sum().sqrt():
                choose_point = indx
            graph[indx]['stamp'] = True
        basis.append(choose_point)
        temp_graph = list(filter(lambda elem: not elem['stamp'], graph))

    return basis


def main(data: Union[str, np.ndarray]):
    # feature, target = exp_sonar()
    # feature, target = exp_real_data2()
    # feature, target = exp_real_data3()
    # feature, target = expe_water()
    # feature, target = exp_airlines()
    # feature, target = wine_example()
    # feature, target = mammonth_example()
    # feature, target = airfoil_exmpl()
    feature = data[:, :-1]
    target = data[:, -1]
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    print(train_feature.shape)

    logging.info("Creating base individ...")
    # base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), graph_file="Info_log\\2024_02_27-12_13_10_PM\\graph_or.txt", n_neighbors=20, eps=0.15)
    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), n_neighbors=20, eps=0.6)
    basis = searсh_basis(base_individ.graph, train_feature)
    with open("test_bas.txt", 'w') as fl:
        fl.write(str(basis))
    other_indiv = DataStructureGraph(train_feature.numpy()[basis], train_target.numpy()[basis], n_neighbors=20, eps=0.15)
    # pass
    base_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=300)
    logging.info("Creating map with operators and population")

    build_settings = {
        'mutation': {
            'simple': dict(intensive=30, increase_prob=1),
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
    runner = ProgramRun()

    boxplot_data = []

    for i in range(1):
        new_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=300)
        result = run_experiment_regression(new_model, test_feature, test_target, i)
        boxplot_data.append(result['f1_score'])

    # runner.save_boxplot("boxplot", boxplot_data)
    runner.save_end_graph(test_target, name="target.txt")


if __name__ == "__main__":
    # data = create_swiss_roll(1000)
    data = create_circle(1, 20)
    # data = "data/electricity-normalized.arff"
    # data = "data/phpSSK7iA.arff"
    # data = "data/sonar_dataset.csv"
    # data = "data/water_potability.csv"
    main(data)


