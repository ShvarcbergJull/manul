import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import scikitplot as skplt
import matplotlib.pyplot as plt

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

def wine_example():
    import pandas as pd
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

    print(len(features))

    return features, target

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

if __name__ == "__main__":
    # feature, target = airfoil_exmpl()
    feature, target = mammonth_example()
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    N = len(test_feature)

    print(len(train_feature), len(test_target))


    # test_target = open(f"C:\\Users\\User\\Desktop\\ntcv\\manul\\results\\mammoth\\target.txt", "r")
    # test_target = np.array(ast.literal_eval(test_target.read()))

    res1 = []
    res2 = []
    val = []

    for i in range(2):
        raw_res = open(f"Info_log\\2024_01_30-01_33_38_PM\\raw_result1_{i}.txt", "r")
        raw_res = raw_res.read().replace("\n", ", ")
        raw_res = np.array(ast.literal_eval(raw_res))
        res = open(f"Info_log\\2024_01_30-01_33_38_PM\\raw_result2_{i}.txt", "r")
        res = res.read().replace("\n", ", ")
        res = np.array(ast.literal_eval(res))

        res1.extend(abs(raw_res.reshape(-1) - test_target.detach().numpy()))
        res2.extend(abs(res.reshape(-1) - test_target.detach().numpy()))
        val.extend([i+1 for _ in range(N)])

    res1 = np.array(res1)
    res2 = np.array(res2)
    import pandas as pd
    df1 = pd.DataFrame({'Experiment': val, 
                        'Prediction Absolute Error': res1, 
                        'Model': 'base'})
    
    df2 = pd.DataFrame({'Experiment': val, 
                        'Prediction Absolute Error': res2, 
                        'Model': 'EA'})
    
    df_aggreg = pd.concat([df1, df2], axis = 0)

    import plotly.express as px
    import plotly.io as pio

    pio.templates.default = "plotly_white"

    px.defaults.template = "plotly_white"
    px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
    px.defaults.width = 600
    px.defaults.height = 500

    fig = px.box(df_aggreg, x="Experiment", y="Prediction Absolute Error", color="Model")
    fig.update_traces(quartilemethod="linear") # or "inclusive", or "linear" by default
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(1, 11)],
            ticktext = [i for i in range(1, 11)]
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline = True, zerolinecolor = 'lightgrey', layer = 'below traces')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline = True, zerolinecolor = 'lightgrey', layer = 'below traces')
    fig.update_layout(
        font_family="Helvetica",
        font_color="black",
        font_size=18,
        title_font_family="Helvetica",
        title_font_color="black",
        title_font_size=18,
        legend_title_font_family='Helvetica',
        legend_title_font_color="black",
        legend_title_font_size=18
    )
    fig.write_image('boxplot_error_mammoth.png', scale=3)

    fig.show()