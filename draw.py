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
    feature, target = wine_example()
    train_feature, train_target, test_feature, test_target, dims = handler_of_data(feature, target)
    N = len(test_feature)

    res1 = []
    res2 = []
    val = []

    for i in range(9):
        raw_res = open(f"C:\\Users\\User\\Desktop\\ntcv\manul\\Info_log\\2024_01_25-02_26_36_PM\\result1_{i}.txt", "r")
        raw_res = np.array(ast.literal_eval(raw_res.read()))
        res = open(f"C:\\Users\\User\\Desktop\\ntcv\manul\\Info_log\\2024_01_25-02_26_36_PM\\result2_{i}.txt", "r")
        res = np.array(ast.literal_eval(res.read()))

        res1.extend(abs(raw_res.reshape(-1) - test_target.detach().numpy()))
        res2.extend(abs(res.reshape(-1) - test_target.detach().numpy()))
        val.extend([i+1 for _ in range(N)])

    res1 = np.array(res1)
    res2 = np.array(res2)
    import pandas as pd
    df1 = pd.DataFrame({'num': val, 
                        'value': res1, 
                        'type': 'start'})
    
    df2 = pd.DataFrame({'num': val, 
                        'value': res2, 
                        'type': 'end'})
    
    df_aggreg = pd.concat([df1, df2], axis = 0)

    import plotly.express as px
    import plotly.io as pio

    pio.templates.default = "plotly_white"

    px.defaults.template = "plotly_white"
    px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
    px.defaults.width = 600
    px.defaults.height = 500

    fig = px.box(df_aggreg, x="num", y="value", color="type")
    fig.update_traces(quartilemethod="linear") # or "inclusive", or "linear" by default
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(1, 10)],
            ticktext = [i for i in range(1, 10)]
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
    fig.write_image('SG_sharepart_ode.png', scale=3)

    fig.show()

    