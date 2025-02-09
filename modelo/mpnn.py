import torch
import torch.nn as nn #modulo que cria as camadas das redes Neurais
import torch.nn.functional as F # contem funções como a de ativação e perda
from torch_geometric.nn import MessagePassing #classe para criar redes baseadas em ássagem de mensagens

class MPNN(MessagePassing):
    def __init__(self, in_features, out_features):
        super(MPNN, self).__init__(aggr='mean')  # Agregação por média
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin(x_j)
