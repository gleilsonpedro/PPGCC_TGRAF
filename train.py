import torch
import torch.nn.functional as F
from utils.data_loader import load_dataset
from modelo import GCN, GAT, SGC, MPNN

# Definir qual dataset usar
DATASET_NAME = "Cora"  # Opções: "Cora", "Citeseer", "PubMed"
MODEL_NAME = "GCN"  # Opções: "GCN", "GAT", "SGC", "MPNN"

# Carregar o dataset
data = load_dataset(DATASET_NAME)
num_features = data.x.shape[1]
num_classes = len(set(data.y.tolist()))

# Definir dimensão oculta
HIDDEN_DIM = 16  # Dimensão intermediária para modelos que usam

# Escolher modelo
if MODEL_NAME == "GCN":
    model = GCN(num_features, HIDDEN_DIM, num_classes)
elif MODEL_NAME == "GAT":
    model = GAT(num_features, HIDDEN_DIM, num_classes)
elif MODEL_NAME == "SGC":
    model = SGC(num_features, num_classes)  # Sem HIDDEN_DIM
elif MODEL_NAME == "MPNN":
    model = MPNN(num_features, num_classes)  # Sem HIDDEN_DIM
else:
    raise ValueError("Modelo inválido! Escolha entre 'GCN', 'GAT', 'SGC' ou 'MPNN'.")


# Configurar otimizador
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Treinamento
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Avaliação
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# Loop de treinamento
EPOCHS = 200
for epoch in range(EPOCHS):
    loss = train()
    if epoch % 10 == 0:
        acc = evaluate()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

print("Treinamento finalizado!")
