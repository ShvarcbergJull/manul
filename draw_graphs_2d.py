import numpy as np
import torch
import logging
import ast
import os
import sys

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

from base.entities import IsolateGraph

logger = logging.getLogger()

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

def wine_example():
    import pandas as pd
    logger.info("Data with wine")
    df = pd.read_csv("data/winequality-red.csv")
    features = df[df.keys()[:-1]].to_numpy()
    target = df[df.keys()[-1]].to_numpy()

    return features, target


if __name__ == "__main__":
    feature, target = wine_example()
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    graph = open("")
    graph = ast.literal_eval(graph)

    my_object = IsolateGraph(data=train_feature, colors=train_target, graph=graph)
    index_point = IsolateGraph.get_started_point(graph_neigh=graph)

    my_object.structure[index_point]['min_distance'] = 0
    my_object.structure[index_point]['from_node'] = None
    my_object.dijkstra([index_point])

    