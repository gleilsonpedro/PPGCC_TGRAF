import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
from utils.data_loader import load_dataset

def train(model, dataset, epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Treina um modelo de aprendizado de grafos em um dataset específico.

    Args:
        model (torch.nn.Module): O modelo a ser treinado (GAT, GCN, SGC, MPNN).
        dataset (torch_geometric.data.Dataset): O dataset a ser usado (Cora, Citeseer ou PubMed).
        epochs (int): Número de épocas de treinamento.
        lr (float): Taxa de aprendizado.
        weight_decay (float): Penalização L2 para regularização.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = dataset[0].to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Avaliação durante o treinamento
        train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
        val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])

        print(f'Época {epoch+1}: Perda = {loss:.4f}, Treino Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}')

def accuracy(pred, labels):
    """Calcula a acurácia das previsões."""
    return (pred.argmax(dim=1) == labels).sum().item() / labels.size(0)

if __name__ == "__main__":
    from models.gcn import GCN  # Exemplo com GCN

    dataset = load_dataset("Cora")
    model = GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes)
    
    train(model, dataset)
