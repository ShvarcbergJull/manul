import numpy as np

def exp_real_data2():
    from scipy.io.arff import loadarff
    import pandas as pd
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

def exp_real_data3():
    from scipy.io.arff import loadarff
    raw_data = loadarff("data/phpSSK7iA.arff")
    df_data = pd.DataFrame(raw_data[0])
    target = df_data['target']
    target[target==b'1'] = 1
    target[target==b'0'] = 0
    target = target.astype(int)

    ks = list(df_data.keys())
    ks = ks[:-1]
    feature = df_data[ks]

    return feature, target

def exp_sonar():
    import csv
    rows = []
    with open('data/sonar_dataset.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            rows.append(list(row[0].split(',')))
    rows = np.array(rows)
    target_old = rows[:, -1]
    features_old = rows[:, :-1]

    datar = features_old[target_old == "R"]
    datam = features_old[target_old == "M"]

    targetr = target_old[target_old == "R"]
    targetm = target_old[target_old == "M"]

    features = []
    target= []

    num = np.max([len(datar), len(datam)])
    for i in range(num):
        try:
            val1 = datar[i]
        except:
            features.extend(datam[i:])
            target.extend(targetm[i:])
            break

        try:
            val2 = datam[i]
        except:
            features.extend(datar[i:])
            target.extend(targetr[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(targetr[i])
        target.append(targetm[i])

    target = np.array(target)
    features = np.array(features)

    target[target == "R"] = 0
    target[target == "M"] = 1

    features = features.astype("float64")
    target = target.astype("int64")

    return features, target


def expe_water():
    import csv
    rows = []
    with open("data/water_potability.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            rows.append(list(row[0].split(',')))
    rows = np.array(rows)
    rows = rows[1:, [1,2,3,5,6,8,9]]

    target_old = rows[:, -1].astype("int64")
    features_old = rows[:, :-1].astype("float64")

    data_0 = features_old[target_old == 0]
    data_1 = features_old[target_old == 1]

    target_0 = target_old[target_old == 0]
    target_1 = target_old[target_old == 1]

    features = []
    target= []

    num = np.max([len(data_0), len(data_1)])
    for i in range(num):
        try:
            val1 = data_0[i]
        except:
            # features.extend(data_1[i:])
            # target.extend(target_1[i:])
            break

        try:
            val2 = data_1[i]
        except:
            # features.extend(data_0[i:])
            # target.extend(target_0[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(target_0[i])
        target.append(target_1[i])

    target = np.array(target)
    features = np.array(features)

    return features, target

def exp_airlines():
    import pandas as pd
    df = pd.read_csv("data/airlines_delay.csv")
    unique_lines = df["Airline"].unique()
    unique_ports = df["AirportFrom"].unique()

    ar = np.array(df["Airline"])
    dr = np.array(df["AirportFrom"])
    fr = np.array(df["AirportTo"])

    for i, value in enumerate(unique_lines):
        ar[ar == value] = i

    for i, value in enumerate(unique_ports):
        dr[dr == value] = i
        fr[fr == value] = i

    ar = np.array([ar, dr, fr])

    features = df[["Flight", "Time", "Length", "DayOfWeek"]].to_numpy()
    features = np.hstack([features, ar.T])
    target = df['Class'].to_numpy()

    target_old = target.astype("int64")
    features_old = features.astype("float64")

    data_0 = features_old[target_old == 0]
    data_1 = features_old[target_old == 1]

    target_0 = target_old[target_old == 0]
    target_1 = target_old[target_old == 1]

    features = []
    target= []

    num = np.max([len(data_0), len(data_1)])
    for i in range(num):
        try:
            val1 = data_0[i]
        except:
            features.extend(data_1[i:])
            target.extend(target_1[i:])
            break

        try:
            val2 = data_1[i]
        except:
            features.extend(data_0[i:])
            target.extend(target_0[i:])
            break

        features.append(val1)
        features.append(val2)

        target.append(target_0[i])
        target.append(target_1[i])

    target = np.array(target)
    features = np.array(features)

    return features[:6000], target[:6000]

def wine_example():
    import pandas as pd
    df = pd.read_csv("data/winequality-red.csv")
    features = df[df.keys()[:-1]].to_numpy()
    target = df[df.keys()[-1]].to_numpy()

    return features, target

def airfoil_exmpl():
    import pandas as pd
    df = pd.read_csv("data/AirfoilSelfNoise.csv")
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

    data = np.array(sorted(data, key=lambda parameters: parameters[1]))
    new_data = []
    new_colors = []

    for i, dt in enumerate(data):
        # if i % 3 != 0:
        #     continue
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