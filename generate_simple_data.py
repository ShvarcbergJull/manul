from sklearn.datasets import make_swiss_roll
from scipy.io.arff import loadarff

import numpy as np
import pandas as pd

def create_circle(radius, N):
    theta = np.random.random(size=N) * 2 * np.pi
    phi = np.random.random(size=N) * np.pi

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    colors = np.linspace(0, 0.9, N)

    sorted_data = np.array([x, y, z])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]

    return np.array([x, y, z, colors]).T

def create_swiss_roll(N):
    data = make_swiss_roll(n_samples=N)

    x = data[0][:, 0]
    y = data[0][:, 1]
    z = data[0][:, 2]
    colors = data[1].copy()
    colors[colors<10] = 0
    colors[colors>=10] = 1

    return np.array([x, y, z, colors]).T

def create_cylindr(radius, N):
    theta = np.random.random(size=N) * 2 * np.pi
    z = np.random.random(size=N)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    min_x = np.min(x)
    min_y = np.min(y)

    colors = np.linspace(0, 0.9, N)
    
    sorted_data = np.array([x, y, z])
    # sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[1]/np.sqrt(parameters[0] ** 2 + parameters[1] ** 2)))
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: [parameters[0], parameters[1], parameters[2]]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]

    return np.array([x, y, z, colors]).T

def create_hepta():
    raw_data = loadarff("data/atom.arff")
    df_data = pd.DataFrame(raw_data[0])

    each_color = np.unique(df_data["class"])
    all_colors = np.linspace(0, 0.9, len(each_color))

    colors = df_data["class"].to_numpy()
    for i, elem in enumerate(each_color):
        colors[colors==elem] = all_colors[i]

    x = df_data["x"]
    y = df_data["y"]
    z = df_data["z"]

    return np.array([x, y, z, colors]).T