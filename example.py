import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors_tool
import time

from fastnet import Graph
import base.methods as mth

if __name__ == "__main__":
    R = 5
    n = 1000

    dn = 2

    theta = np.random.random(size=n) * 2 * np.pi
    phi = np.random.random(size=n) * np.pi

    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)

    colormap = cm.viridis
    colors = [colors_tool.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, n)]

    sorted_data = np.array([x, y, z])
    sorted_data = np.array(sorted(sorted_data.T, key=lambda parameters: parameters[2]))

    x = sorted_data[:, 0]
    y = sorted_data[:, 1]
    z = sorted_data[:, 2]

    fir = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(x, y, z, c=colors)

    plt.show()

    data = np.array([x, y, z]).T

    data, avg_of_data, var_of_data = mth.prebording_data(data)
    dataN = len(data)
    eps = 0.3

    time_start = time.time()
    graph = Graph(dataN, data.tolist(), colors, avg_of_data, var_of_data, eps)
    print("--- %s seconds ---" % (time.time() - time_start))
    drawing = mth.Draw(graph)

    # drawing.draw_graph()

    choose_node = None
    for node in graph.nodes():
        if not choose_node:
            choose_node = node
        elif len(node.neighbours()) > len(choose_node.neighbours()):
            choose_node = node

    graph.check_visible_neigh(choose_node)

    drawing.draw_graph()
