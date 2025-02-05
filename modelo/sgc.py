import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SGC(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGC, self).__init__()
        self.conv = SGConv(in_features, out_features, K=2)  # K = número de hops

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
