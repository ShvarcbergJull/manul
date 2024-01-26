import numpy as np
import torch
import logging
import ast
import os
import sys
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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
        if i % 3 != 0:
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


def airfoil_exmpl():
    import pandas as pd
    df = pd.read_csv("data/AirfoilSelfNoise.csv")
    features = df[df.keys()[:-1]].to_numpy()
    target = df[df.keys()[-1]].to_numpy()

    return features, target

def draw(graph: IsolateGraph):
    edges=[]
    for edge in graph.structure:
        pos1 = graph.structure[edge]['orig_pos']
        for neig in graph.structure[edge]['neighbours']:
            edges.append(pos1)
            edges.append(graph.structure[neig]['orig_pos'])
            edges.append([None, None, None])
    
    edges = np.array(edges).T
    edge_trace = go.Scatter3d(x=edges[0], y=edges[1], z=edges[2], line=dict(width=4, color='#888'), hoverinfo='none', mode='lines')
    
    nodes = np.array([graph.structure[node]['orig_pos'] for node in graph.structure]).T
    colors = np.array([graph.structure[node]['marker'] for node in graph.structure])
    node_trace = go.Scatter3d(x=nodes[0], y=nodes[1], z=nodes[2], mode='markers', hoverinfo='text',
                                marker=dict(
                                    showscale=True,
                                    # colorscale options
                                    # #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                    # #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                    # #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                                    colorscale='YlGnBu',
                                    reversescale=True,
                                    color=colors,
                                    size=10,
                                colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                titleside='right'
                                ),
                                line_width=2))
    
    fig = go.Figure(data=[node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()


if __name__ == "__main__":
    feature, target = airfoil_exmpl()
    # feature, target = mammonth_example()
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    graph = open("Info_log\\2024_01_26-02_13_46_PM\\graph.txt", "r")
    graph = ast.literal_eval(graph.read())

    my_object = IsolateGraph(data=train_feature.detach().numpy(), colors=train_target.detach().numpy(), graph=graph)
    index_point = IsolateGraph.get_started_point(graph_neigh=graph)

    my_object.structure[index_point]['min_distance'] = 0
    my_object.structure[index_point]['from_node'] = None
    my_object.dijkstra([index_point])

    print("PCA")
    fit_data = my_object.get_data_for_pca(index_point)
    pca = PCA(n_components=3)
    pca.fit(fit_data)
    result = pca.transform(fit_data)
    my_object.set_new_params(index_point, result)

    my_object.find_raw_params(pca)
    nodes = my_object.structure[index_point]['neighbours']
    print("TRANSFORM")
    other_nodes = my_object.transform_nodes(nodes)

    draw(my_object)