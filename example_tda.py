# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module
import matplotlib.cm as cm
import matplotlib.colors as colors_tool
from scipy.io.arff import loadarff
from sklearn.datasets import load_breast_cancer

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA


import base.methods as mth


if __name__ == "__main__":
    # R = 5
    # n = 1000

    # dn = 1

    # theta = np.random.random(size=n) * 2 * np.pi
    # phi = np.random.random(size=n) * np.pi

    # x = R * np.cos(theta) * np.sin(phi)
    # y = R * np.sin(theta) * np.sin(phi)
    # z = R * np.cos(phi)

    # colormap = cm.viridis
    # colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]

    # sorted_data = np.array([x, y, z])
    # sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    # x = sorted_data[:, 0]
    # y = sorted_data[:, 1]
    # z = sorted_data[:, 2]

    # data = np.array([x, y, z]).T


    # real data
    dn = 1

    data = load_breast_cancer(return_X_y=True, as_frame=True)
    feature = data[0]
    target = data[1]


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

    raw_data = loadarff("data/phpSSK7iA.arff")
    df_data = pd.DataFrame(raw_data[0])
    target = df_data['target']
    target[target==b'1'] = 1
    target[target==b'0'] = 0
    target = target.astype(int)

    ks = list(df_data.keys())
    ks = ks[:-1]
    feature = df_data[ks]

    data, avg_of_data, var_of_data = mth.prebording_data(feature.values)

    indexs_for_project = np.argsort(var_of_data)

    filter_func = Projection(columns=list(indexs_for_project[-3:]))
    cover = CubicalCover(n_intervals=4, overlap_frac=0.3)
    clusterer = DBSCAN()
    # clusterer = AffinityPropagation()

    n_jobs = 1

    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=True,
        n_jobs=n_jobs,
    )

    result = pipe.fit_transform(data, target)

    fig = plot_static_mapper_graph(pipe, data)
    fig.show(config={'scrollZoom': True})
