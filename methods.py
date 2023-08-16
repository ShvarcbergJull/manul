# File with methods for  various operations

from numba import njit
import numpy as np
import torch
from sklearn.metrics import roc_curve

@njit
def distance(p, q, matrix):
    '''
    Searching the Euclidean distance between 2 points in the input matrix
    '''
    sm = 0
    for key in range(matrix.shape[-1]):
        dsq = (matrix[p][key] - matrix[q][key]) ** 2
        sm += dsq
    return np.sqrt(sm)

@njit
def find_manifold_loss(f_data, f_x):
    '''
    - seraching the Euclidean distance between all pair points
    - normalizing matric od distances
    - searching laplasian matrix (L = A - D)
    - calculating loss (X^TLX)
    '''
    N = len(f_data)
    graph = np.zeros((N, N))
    # print(graph.shape)
    for i in range(N):
        for j in range(i, N):
            val = distance(i, j, f_data)
            graph[i][j] = val
            graph[j][i] = val

    graph_new = graph/graph.max()
    adjacency_matrix = np.ones((N, N))
    for i in range(N):
        adjacency_matrix[i][i] = 0
    laplasian = adjacency_matrix - graph_new

    part_1 = np.dot(f_x.T, laplasian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]

@njit
def find_ED(f_data, eps):
    '''
    Seraching the Euclidean distance between all pair points
    '''
    N = len(f_data)
    graph = np.zeros((N, N))
    # print(graph.shape)
    for i in range(N):
        for j in range(i, N):
            val = distance(i, j, f_data)
            graph[i][j] = val
            graph[j][i] = val

    print(graph.max())

    graph_new = graph/graph.max()
    # graph_new = np.putmask(graph_new, graph_new>eps, 0)
    
    return graph_new

@njit
def count_neighbors(graph):
    all_number = graph.shape[1]
    result = []
    max_num = 0
    id_max = None
    for i in range(graph.shape[0]):
        number = np.sum(graph[i]==0)
        number = all_number - number
        if number > max_num:
            id_max = i
            max_num = number
        # print(number, id_max)
        result.append(number)
    
    return result, id_max

@njit
def point_with_neighbors(data, id_point, graph):
    neighbors = graph[id_point]
    new_data = []
    for j, neigh in enumerate(neighbors):
        if neigh > 0 or j == id_point:
            new_data.append(data[j])

    return new_data





def split_data_TT(feature, target):
    grid_tensors = [torch.tensor(feature[key].values) for key in feature.keys()]
    grid_tensor = torch.stack(grid_tensors)
    grid_flattened = grid_tensor.view(grid_tensor.shape[0], -1).transpose(0, 1)
    grid_flattened = grid_flattened.to(grid_flattened.to(torch.float64))
    grid_flattened[0]

    train_features = grid_flattened
    train_target = target

    param = len(target) // 100 * 80
    train_features = grid_flattened[:param]
    train_target = target[:param]
    test_features = grid_flattened[param:]
    test_target = target[param:]

    try:
        train_target = torch.tensor(train_target)
    except ValueError:
        train_target = torch.tensor(train_target.to_numpy())
    try:
        test_target = torch.tensor(test_target)
    except ValueError:
        test_target = torch.tensor(test_target.to_numpy())

    print("TRAIN:", len(train_target))
    print("TEST:", len(test_target))

    return train_features, train_target, test_features, test_target

def fit_nn(train_features, train_target, model, optimizer, criterion, manifold=False, **kwargs):
    model.train()
    lmd = kwargs.get("lmd")
    num_epochs = kwargs.get("num_epochs", 500)
    batch_size = kwargs.get("batch_size", 750)
    threshold = None
    min_loss, t = np.inf, 0
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_features.size()[0])
        loss_list = []
        for i in range(0, len(train_target), batch_size):
            indices = permutation[i:i+batch_size]
            # print(indices)
            batch_x, target_y = train_features[indices], train_target[indices]
            target_y = target_y.to(torch.float64)
            optimizer.zero_grad()
            output = model(batch_x)
            add_loss = None
            if manifold:
                add_loss = find_manifold_loss(batch_x.numpy(), output.detach().numpy())
            loss = criterion(output, target_y.reshape_as(output))
            fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            if not threshold:
                threshold = thresholds[ix]
            else:
                threshold = np.mean([thresholds[ix], threshold])
            
            if manifold and add_loss:
                loss += lmd * add_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        loss_mean = np.mean(loss_list)

        t += 1
        print('Surface training t={}, loss={}'.format(t, loss_mean))

    model.eval()
    return model

