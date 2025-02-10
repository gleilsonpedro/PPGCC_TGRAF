import time
import torch
import torch.nn.functional as F
from utils.data_loader import load_dataset
from modelo import MPNN

# Função para treinar o MPNN e exibir loss e acurácia por época
def train_mpnn(dataset_name, epochs=200, lr=0.01):
    print(f'Treinando MPNN no dataset {dataset_name}...')
    
    # Carregar dataset
    data = load_dataset(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPNN(data.num_features, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Avaliação
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            acc = correct / data.test_mask.sum().item()
        
        # Exibir logs a cada 10 épocas
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Dataset: {dataset_name} | Modelo: MPNN | Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}')
    
    total_time = time.time() - start_time
    print(f'🏁 {dataset_name} - Tempo total: {total_time:.2f}s | Loss final: {loss.item():.4f} | Acurácia final: {acc:.4f}\n')
    
    return total_time, loss.item(), acc

if __name__ == "__main__":
    datasets = ["Cora", "Citeseer", "PubMed"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = train_mpnn(dataset)
    
    # Exibir resumo final
    print("📊 Resumo Final:")
    for dataset, (time_taken, final_loss, final_acc) in results.items():
        print(f'Dataset: {dataset} | Tempo: {time_taken:.2f}s | Loss Final: {final_loss:.4f} | Acurácia Final: {final_acc:.4f}')
