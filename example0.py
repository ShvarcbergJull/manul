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

from base.entities import DataStructureGraph, PopulationGraph
from base.operators.builder import create_operator_map

from generate_simple_data import create_swiss_roll

def handler_of_data(data: Union[str, np.ndarray]):
    if type(data) is str:
        format_doc = data.split(".")[-1]
        if format_doc == "arff":
            from scipy.io.arff import loadarff
            raw_data = pd.DataFrame(loadarff(data))
            target_key = raw_data.keys()[-1]
            feature_keys = raw_data.keys()[:-1]

            target = raw_data[target_key]
            for i, elem in target.unique():
                target[target==elem] = i
            target = target.astype(int)

            feature = raw_data[feature_keys]
    else:
        feature = data[:, :-1]
        target = data[:, -1]

    try:
        grid_tensors = [torch.tensor(feature[key].values) for key in feature.keys()]
        grid_tensor = torch.stack(grid_tensors)
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



def main(data: Union[str, np.ndarray]):
    train_feature, train_target, test_feature, test_target = handler_of_data(data=data)

    logging.info("Creating base individ...")
    base_individ = DataStructureGraph(train_feature.numpy(), train_target.numpy(), eps=0.15)
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
        }
    }

    create_operator_map(train_feature, base_individ, build_settings)


    population = PopulationGraph(iterations=2)
    population.evolutionary()


if __name__ == "__main__":
    data = create_swiss_roll(500)
    main(data)


