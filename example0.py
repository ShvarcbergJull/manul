import os
import sys
from typing import Union
import torch

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from base.entities import DataStructureGraph, PopulationGraph, TakeNN
from base.operators.builder import create_operator_map

from generate_simple_data import create_swiss_roll

def handler_of_data(data: Union[str, np.ndarray]):
    if type(data) is str:
        format_doc = data.split(".")[-1]
        if format_doc == "arff":
            from scipy.io.arff import loadarff
            raw_data = pd.DataFrame(loadarff(data)[0])
            raw_data['day'] = raw_data['day'].astype('int32')
            target_key = raw_data.keys()[-1]
            feature_keys = raw_data.keys()[:-1]

            target = raw_data[target_key]
            for i, elem in enumerate(np.unique(target.to_numpy())):
                target[target==elem] = i
            target = target.astype(int)

            feature = raw_data[feature_keys]
            # feature = feature.to_numpy()
    else:
        feature = data[:, :-1]
        target = data[:, -1]

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

    return train_features, train_target, test_features, test_target

def find_graph_loss(graph, f_x, indexs=None):
    if indexs is None:
        laplassian = graph.laplassian
        
    else:
        laplassian = graph.laplassian[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]


def main(data: Union[str, np.ndarray]):
    train_feature, train_target, test_feature, test_target = handler_of_data(data=data)
    print(train_feature.shape)

    logging.info("Creating base individ...")
    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), n_neighbors=20, eps=0.3)
    base_model = TakeNN(train_feature, train_target, dims=train_feature.numpy().shape[1], num_epochs=30, batch_size=300)
    logging.info("Creating map with operators and population")

    build_settings = {
        'mutation': {
            'simple': dict(intensive=2, increase_prob=1),
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

    population = PopulationGraph(iterations=30)
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


if __name__ == "__main__":
    # data = create_swiss_roll(3000)
    data = "data/electricity-normalized.arff"
    # data = "data/phpSSK7iA.arff"
    main(data)


