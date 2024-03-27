import os
import sys
from typing import Union
import torch

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

import numpy as np
from numba import njit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_squared_error
import matplotlib.pyplot as plt

from base.entities import DataStructureGraph, PopulationGraph, TakeNN
from base.operators.builder import create_operator_map
from base.operators.base import ProgramRun

from generate_simple_data import create_swiss_roll, create_circle
from data_forming import airfoil_exmpl, exp_airlines, mammonth_example, exp_real_data2

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


def run_experiment(base_model, test_feature, test_target, number):
    runner = ProgramRun()

    base_model.train() # обучение базовой модели
    result1 = base_model.model_settings['model'](test_feature) # получение результатов из модели на тестовой выборке
    result1 = result1.detach().numpy()
    if base_model.problem == 'class':
        result1 = np.where(result1 > base_model.threshold, 1, 0)
        runner.save_pickle(result1, f"result1_{number}.pkl") # сохранение результата
        metric_nn_1 = f1_score(test_target.reshape(-1), result1.reshape(-1), average=None) # считается метрика
    else:
        runner.save_pickle(result1, f"raw_result1_{number}.pkl")
        result1 = result1.round().astype("int64")
        runner.save_pickle(result1, f"result1_{number}.pkl")
        metric_nn_1 = mean_squared_error(test_target.reshape(-1), result1.reshape(-1))


    population = PopulationGraph(iterations=3) # создание популяции
    population.evolutionary(num=number) # запуск ЭА

    result2 = population.base_model.model_settings['model'](test_feature) # получение результатов на модели из ЭА на тестовой выборке
    result2 = result2.detach().numpy()
    if base_model.problem == 'class':
        result2 = np.where(result2 > population.base_model.threshold, 1, 0)
        runner.save_pickle(result2, f"result2_{number}.pkl")
        metric_nn_2 = f1_score(test_target.reshape(-1), result2.reshape(-1), average=None)
        runner.save_confusion_matrix(f"conf_just_model_{number}", data=[test_target, result1])
        runner.save_confusion_matrix(f"conf_EA_model_{number}", data=[test_target, result2])
    else:
        runner.save_pickle(result2, f"raw_result2_{number}.pkl")
        result2 = result2.round().astype("int64")
        runner.save_pickle(result2, f"result2_{number}.pkl")
        metric_nn_2 = mean_squared_error(test_target.reshape(-1), result2.reshape(-1))

    runner.save_plot(f"fitness_{number}", population.change_fitness)
    runner.save_model(f"model_{number}", population.base_model.model_settings['model'])

    return_dictionary = {
        "f1_score": [list(metric_nn_1), list(metric_nn_2)],
    }

    return return_dictionary


def main(feature, target):
    
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target) # разделение данных на обуяающую и тестовую выборки

    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), n_neighbors=10, eps=0.15, mode=0) # создание базового индивида
    with open("test_bas.txt", 'w') as fl:
        fl.write(str(base_individ.basis)) # сохранение индексов точек от всех данных, нужно для рисование результата
    base_model = TakeNN(train_feature[base_individ.basis], train_target[base_individ.basis], dims=dims, num_epochs=30, batch_size=300, problem='regres') # создание модели

    # словарик настроек для жволюционных операторов
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
            'add_loss_function': find_graph_loss # функция, которая используется в процессе обучения модели с графом
        }
    }

    create_operator_map(train_feature, base_individ, base_model.copy(), build_settings) # создание OperatorMap
    runner = ProgramRun() # просто для инициализации

    boxplot_data = []

    for i in range(1): # запуск цикла по кол-ву запусков эксперимента
        new_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=300, problem='regres') # создаybt базовой модели (на которой не будет применяться обучения с учётом графа) 
        result = run_experiment(new_model, test_feature, test_target, i) # запуск эксперимента
        # boxplot_data.append(result['f1_score'])

    runner.save_end_graph(test_target, name="target.txt")


if __name__ == "__main__":
    # вызываются данные

    # data = create_swiss_roll(2000)
    # feature = data[:, :-1]
    # target = data[:, -1]

    # feature, target = exp_sonar()
    # feature, target = exp_real_data2()
    # feature, target = exp_real_data3()
    # feature, target = expe_water()
    # feature, target = exp_airlines()
    # feature, target = wine_example()
    feature, target = mammonth_example()
    # feature, target = airfoil_exmpl()
    main(feature, target)


