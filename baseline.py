import torch
import torch.nn as nn
import pandas as pd
from scipy.io.arff import loadarff


def baseline(dim):
    baseline_model = nn.Sequential(
        nn.Linear(dim, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    return baseline_model


if __name__ == "__main__":
    raw_data = loadarff("data/phpkIxskf.arff")
    df_data = pd.DataFrame(raw_data[0])

    df_data

