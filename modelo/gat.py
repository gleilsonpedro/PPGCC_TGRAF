import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden_features, heads=heads)
        self.conv2 = GATConv(hidden_features * heads, out_features, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
