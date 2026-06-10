import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv

from model.hydro_net import PEM
from model.model_cfg import CFG
# import matplotlib.pyplot as plt


class SupervisedEnergyRegressorHead(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

