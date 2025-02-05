import torch
import torch.nn.functional as F
from utils.data_loader import load_dataset
from modelo import GCN, GAT, SGC, MPNN
import pandas as pd

# Modelos disponíveis
MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "SGC": SGC,
    "MPNN": MPNN
}

# Datasets disponíveis
DATASETS = ["Cora", "Citeseer", "PubMed"]

# Configuração do treinamento
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 16  # Dimensão da camada oculta

# Lista para armazenar resultados
results = []

for dataset_name in DATASETS:
    # Carregar dataset
    data = load_dataset(dataset_name)
    num_features = data.x.shape[1]
    num_classes = len(set(data.y.tolist()))
    
    for model_name, ModelClass in MODELS.items():
        # Inicializar modelo
        if model_name == "SGC":
            model = ModelClass(num_features, num_classes)  # SGC não tem hidden layer
        else:
            model = ModelClass(num_features, HIDDEN_DIM, num_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()
        
        def evaluate():
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            return acc
        
        print(f"Treinando {model_name} no dataset {dataset_name}...")
        for epoch in range(EPOCHS):
            loss = train()
            if epoch % 10 == 0:
                acc = evaluate()
                print(f"Dataset: {dataset_name} | Modelo: {model_name} | Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")
        
        # Avaliação final
        final_acc = evaluate()
        results.append([dataset_name, model_name, final_acc])

# Salvar os resultados em um DataFrame
results_df = pd.DataFrame(results, columns=["Dataset", "Modelo", "Acurácia"])
print(results_df)

# Salvar em CSV para análise posterior
results_df.to_csv("comparacao_resultados.csv", index=False)
