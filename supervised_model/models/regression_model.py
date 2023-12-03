import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

from model.hydro_net import PEM
from model.model_cfg import CFG
# import matplotlib.pyplot as plt


class EnergyRegressionModel(nn.Module):
    def __init__(self, feature_extractor, regressor):
        super(EnergyRegressionModel, self).__init__()
        self.feature_extractor = feature_extractor()
        self.regressor = regressor()

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output
