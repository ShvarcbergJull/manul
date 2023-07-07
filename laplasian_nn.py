from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

@njit
def distance(p, q, matrix):
    sm = 0
    for key in range(matrix.shape[-1]):
        dsq = (matrix[p][key] - matrix[q][key]) ** 2
        sm += dsq
    return np.sqrt(sm)

@njit
def find_manifold_loss(f_data, f_x):
    N = len(f_data)
    graph = np.zeros((N, N))
    # print(graph.shape)
    for i in range(N):
        for j in range(i, N):
            val = distance(i, j, f_data)
            graph[i][j] = val
            graph[j][i] = val

    # print(graph.shape)

    graph_new = graph/graph.max()
    # graph_new = graph
    adjacency_matrix = np.ones((N, N))
    for i in range(N):
        adjacency_matrix[i][i] = 0
    # adjacency_matrix = np.zeros((N, N))
    # for i in range(N):
    #     adjacency_matrix[i][i] = N
    laplasian = adjacency_matrix - graph_new

    # part_1 = np.dot(f_x.detach().numpy().T, laplasian)
    part_1 = np.dot(f_x.T, laplasian)
    loss = np.dot(part_1, f_x)

    return loss.reshape(-1)[0]

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


# simple data
# data = load_breast_cancer(return_X_y=True, as_frame=True)
# feature = data[0]
# target = data[1]
# ------------------

# real_data
'''
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
'''

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


dims = len(feature.keys())
b_model = baseline(dims)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(b_model.parameters(), lr=1e-4, eps=1e-4)
b_model.train()

x_model = baseline(dims)
criterion = nn.BCELoss()
x_optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-4, eps=1e-4)
x_model.train()

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

train_target = torch.tensor(train_target)
test_target = torch.tensor(test_target)

# print(len(train_target[train_target==0]), len(train_target[train_target==1]))
# print(len(test_target[test_target==0]), len(test_target[test_target==1]))
# exit(1)

print("TRAIN:", len(train_target))
print("TEST:", len(test_target))

print("TRAIN 1")

batch_size = 750
# num_epochs = 2000
num_epochs = 500
min_loss, t = np.inf, 0
threshold = None
for epoch in range(num_epochs):
    permutation = torch.randperm(train_features.size()[0])
    loss_list = []
    for i in range(0, len(train_target), batch_size):
        indices = permutation[i:i+batch_size]
        # print(indices)
        batch_x, target_y = train_features[indices], train_target[indices]
        target_y = target_y.to(torch.float64)
        optimizer.zero_grad()
        output = b_model(batch_x)
        # output[output>0.5] = 1
        # output[output<=0.5] = 0
        # print(output.shape)
        loss = criterion(output, target_y.reshape_as(output))
        fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        # print("IX", thresholds[ix])
        if not threshold:
            threshold = thresholds[ix]
        else:
            threshold = np.mean([thresholds[ix], threshold])
        # loss = torch.mean(torch.abs(target_y-output))
        # loss = np.mean(np.abs(output - (target_y.reshape_as(output)).detach().numpy()))
        
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        loss_list.append(loss.item())
    # print(loss_list)
    loss_mean = np.mean(loss_list)

    t += 1
    print('Surface training t={}, loss={}'.format(t, loss_mean))

b_model.eval()
# baseline_out = b_model(train_features)
# baseline_out = baseline_out.detach().numpy()

# fpr, tpr, thresholds = roc_curve(train_target.reshape(-1), baseline_out.reshape(-1))

# gmeans = np.sqrt(tpr * (1-fpr))
# ix = np.argmax(gmeans)
# threshold = thresholds[ix]

baseline_out = b_model(test_features)
baseline_out = baseline_out.detach().numpy()

# fpr, tpr, thresholds = roc_curve(test_target.reshape(-1), baseline_out.reshape(-1))

# gmeans = np.sqrt(tpr * (1-fpr))
# ix = np.argmax(gmeans)
# print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

# plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# plt.plot(fpr, tpr, marker='.', label='Logistic')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

# baseline_out[baseline_out>0.5] = 1
# baseline_out[baseline_out<=0.5] = 0
baseline_out = np.where(baseline_out > threshold, 1, 0)
# tn, fp, fn, tp = confusion_matrix(test_target.to_numpy(), baseline_out.reshape(-1)).ravel()
cm = confusion_matrix(test_target.reshape(-1), baseline_out.reshape(-1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


print("TRAIN:", len(train_target))
print("TEST:", len(test_target))
print("TRAIN 2")

batch_size = 750
# num_epochs = 2000
num_epochs = 500
min_loss, t = np.inf, 0
val = np.min([batch_size, len(feature)])
lmd = 1/(val ** 2)
threshold = None
# lmd = 0.01
for epoch in range(num_epochs):
    permutation = torch.randperm(train_features.size()[0])
    loss_list = []
    lap_list = []
    for i in range(0, len(train_target), batch_size):
        indices = permutation[i:i+batch_size]
        # print(indices)
        batch_x, target_y = train_features[indices], train_target[indices]
        target_y = target_y.to(torch.float64)
        x_optimizer.zero_grad()
        output = x_model(batch_x)
        add_loss = find_manifold_loss(batch_x.numpy(), output.detach().numpy())
        loss = criterion(output, target_y.reshape_as(output))

        fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        if not threshold:
            threshold = thresholds[ix]
        else:
            threshold = np.mean([thresholds[ix], threshold])
        # loss = torch.mean(torch.abs(target_y-output))
        # print(lmd * add_loss)
        # print(loss, lmd*add_loss[0])
        loss += lmd * add_loss
        
        # print(type(loss))
        loss.backward()
        x_optimizer.step()
        # print(loss.item())
        loss_list.append(loss.item())
    # print(loss_list)
    loss_mean = np.mean(loss_list)

    t += 1
    print('Surface training t={}, loss={}'.format(t, loss_mean))

x_model.eval()
# nn_out_s = x_model(train_features)
# nn_out = nn_out_s.detach().numpy()

# fpr, tpr, thresholds = roc_curve(train_target.reshape(-1), nn_out.reshape(-1))

# gmeans = np.sqrt(tpr * (1-fpr))
# ix = np.argmax(gmeans)
# threshold = thresholds[ix]
# print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

# plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# plt.plot(fpr, tpr, marker='.', label='Logistic')
# plt.xlabel('False Positive Rate') 
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

# nn_out[nn_out>0.5] = 1
# nn_out[nn_out<=0.5] = 0

nn_out_s = x_model(test_features)
nn_out = nn_out_s.detach().numpy()
# nn_out[nn_out>0.5] = 1
# nn_out[nn_out<=0.5] = 0
nn_out = np.where(nn_out > threshold, 1, 0)

cm = confusion_matrix(test_target.reshape(-1), nn_out.reshape(-1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()