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

from base.entities import DataStructureGraph, PopulationGraph, TakeNN
from base.operators.builder import create_operator_map
from base.operators.base import ProgramRun

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

def exp_sonar():
    import csv
    rows = []
    with open('data/sonar_dataset.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            rows.append(list(row[0].split(',')))
    rows = np.array(rows)
    target_old = rows[:, -1]
    features_old = rows[:, :-1]

    datar = features_old[target_old == "R"]
    datam = features_old[target_old == "M"]

    targetr = target_old[target_old == "R"]
    targetm = target_old[target_old == "M"]

    features = []
    target= []

    num = np.max([len(datar), len(datam)])
    for i in range(num):
        try:
            val1 = datar[i]
        except:
            features.extend(datam[i:])
            target.extend(targetm[i:])
            break

        try:
            val2 = datam[i]
        except:
            features.extend(datar[i:])
            target.extend(targetr[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(targetr[i])
        target.append(targetm[i])

    target = np.array(target)
    features = np.array(features)

    target[target == "R"] = 0
    target[target == "M"] = 1

    features = features.astype("float64")
    target = target.astype("int64")

    return features, target


def expe_water():
    import csv
    rows = []
    with open("data/water_potability.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            rows.append(list(row[0].split(',')))
    rows = np.array(rows)
    rows = rows[1:, [1,2,3,5,6,8,9]]

    target_old = rows[:, -1].astype("int64")
    features_old = rows[:, :-1].astype("float64")

    data_0 = features_old[target_old == 0]
    data_1 = features_old[target_old == 1]

    target_0 = target_old[target_old == 0]
    target_1 = target_old[target_old == 1]

    features = []
    target= []

    num = np.max([len(data_0), len(data_1)])
    for i in range(num):
        try:
            val1 = data_0[i]
        except:
            # features.extend(data_1[i:])
            # target.extend(target_1[i:])
            break

        try:
            val2 = data_1[i]
        except:
            # features.extend(data_0[i:])
            # target.extend(target_0[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(target_0[i])
        target.append(target_1[i])

    target = np.array(target)
    features = np.array(features)

    return features, target

def exp_airlines():
    import pandas as pd
    df = pd.read_csv("data/airlines_delay.csv")
    unique_lines = df["Airline"].unique()
    unique_ports = df["AirportFrom"].unique()

    ar = np.array(df["Airline"])
    dr = np.array(df["AirportFrom"])
    fr = np.array(df["AirportTo"])

    for i, value in enumerate(unique_lines):
        ar[ar == value] = i

    for i, value in enumerate(unique_ports):
        dr[dr == value] = i
        fr[fr == value] = i

    ar = np.array([ar, dr, fr])

    features = df[["Flight", "Time", "Length", "DayOfWeek"]].to_numpy()
    features = np.hstack([features, ar.T])
    target = df['Class'].to_numpy()

    target_old = target.astype("int64")
    features_old = features.astype("float64")

    data_0 = features_old[target_old == 0]
    data_1 = features_old[target_old == 1]

    target_0 = target_old[target_old == 0]
    target_1 = target_old[target_old == 1]

    features = []
    target= []

    num = np.max([len(data_0), len(data_1)])
    for i in range(num):
        try:
            val1 = data_0[i]
        except:
            features.extend(data_1[i:])
            target.extend(target_1[i:])
            break

        try:
            val2 = data_1[i]
        except:
            features.extend(data_0[i:])
            target.extend(target_0[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(target_0[i])
        target.append(target_1[i])

    target = np.array(target)
    features = np.array(features)

    return features[:6000], target[:6000]

def wine_example():
    import pandas as pd
    df = pd.read_csv("data/winequality-red.csv")
    features = df[df.keys()[:-1]].to_numpy()
    target = df[df.keys()[-1]].to_numpy()

    return features, target

def airfoil_exmpl():
    import pandas as pd
    df = pd.read_csv("data/AirfoilSelfNoise.csv")
    features = df[df.keys()[:-1]].to_numpy()
    target = df[df.keys()[-1]].to_numpy()

    return features, target


def mammonth_example():
    import ast
    fl = open("data/mammoth_3d.json ", "r")
    data = fl.read()
    data = ast.literal_eval(data)

    data = np.array(data)
    N = len(data)
    colors = np.linspace(0, 0.9, N)

    data = np.array(sorted(data, key=lambda parameters: parameters[2]))
    new_data = []
    new_colors = []

    for i, dt in enumerate(data):
        if i % 2 != 0:
            continue
        new_data.append(dt)
        new_colors.append(colors[i])

    data = []
    colors = []

    temp_data = []
    temp_colors = []

    for i, dat in enumerate(new_data):
        if i % 2 != 0:
            temp_data.append(dat)
            temp_colors.append(new_colors[i])
        else:
            data.append(dat)
            colors.append(new_colors[i])

    colors.extend(temp_colors)
    data.extend(temp_data)

    return np.array(data), np.array(colors)


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

    population = PopulationGraph(iterations=15)
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


def main(data: Union[str, np.ndarray]):
    # feature, target = exp_sonar()
    # feature, target = exp_real_data2()
    # feature, target = exp_real_data3()
    # feature, target = expe_water()
    # feature, target = exp_airlines()
    # feature, target = wine_example()
    # feature, target = mammonth_example()
    feature, target = airfoil_exmpl()
    # feature = data[:, :-1]
    # target = data[:, -1]
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    print(train_feature.shape)

    logging.info("Creating base individ...")
    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), n_neighbors=20, eps=0.20)
    base_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=300)
    logging.info("Creating map with operators and population")

    build_settings = {
        'mutation': {
            'simple': dict(intensive=10, increase_prob=1),
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

    for i in range(10):
        new_model = TakeNN(train_feature, train_target, dims=dims, num_epochs=30, batch_size=300)
        result = run_experiment_regression(new_model, test_feature, test_target, i)
        boxplot_data.append(result['f1_score'])

    runner.save_boxplot("boxplot", boxplot_data)
    runner.save_end_graph(test_target, name="target.txt")

    # population = PopulationGraph(iterations=15)
    # population.evolutionary()

    # base_model.train()

    # result1 = base_model.model_settings['model'](test_feature)
    # result1 = result1.detach().numpy()
    # result1 = np.where(result1 > base_model.threshold, 1, 0)

    # result2 = population.base_model.model_settings['model'](test_feature)
    # result2 = result2.detach().numpy()
    # result2 = np.where(result2 > population.base_model.threshold, 1, 0)

    # runner.save_confusion_matrix("conf_just_model", data=[test_target, result1])
    # runner.save_confusion_matrix("conf_EA_model", data=[test_target, result2])


    # cm = confusion_matrix(test_target.reshape(-1), result1.reshape(-1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # # plt.savefig(f"images/{k}_nn")
    # plt.show()

    # cm = confusion_matrix(test_target.reshape(-1), result2.reshape(-1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # # plt.savefig(f"images/{k}_nn")
    # plt.show()

    # metric_nn_1 = f1_score(test_target.reshape(-1), result1.reshape(-1), average=None)
    # metric_nn_2 = f1_score(test_target.reshape(-1), result2.reshape(-1), average=None)

    # with open("example0.txt", "w") as fl:
    #     fl.write(str(list(metric_nn_1)))
    #     fl.write("\n")
    #     fl.write(str(list(metric_nn_2)))


if __name__ == "__main__":
    # data = create_swiss_roll(1000)
    # data = create_circle(5, 5000)
    # data = "data/electricity-normalized.arff"
    # data = "data/phpSSK7iA.arff"
    # data = "data/sonar_dataset.csv"
    data = "data/water_potability.csv"
    main(data)


