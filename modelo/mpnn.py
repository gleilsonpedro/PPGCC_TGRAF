import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class MPNN(MessagePassing):
    def __init__(self, in_features, out_features):
        super(MPNN, self).__init__(aggr='mean')  # Agregação por média
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin(x_j)
