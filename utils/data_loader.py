import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_dataset(name, root='./data'):
    """
    Carrega um dos datasets Planetoid (Cora, Citeseer ou PubMed).

    Args:
        name (str): Nome do dataset ('Cora', 'Citeseer' ou 'PubMed').
        root (str): Diretório onde os dados serão armazenados.

    Returns:
        torch_geometric.data.Dataset: Dataset carregado.
    """
    dataset = Planetoid(root=f'{root}/{name}', name=name, transform=NormalizeFeatures())
    return dataset

if __name__ == "__main__":
    # Teste para garantir que os datasets são carregados corretamente
    for dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = load_dataset(dataset_name)
        print(f"Dataset {dataset_name} carregado com sucesso!")
        print(f"Número de nós: {dataset[0].num_nodes}")
        print(f"Número de arestas: {dataset[0].num_edges}")
        print(f"Número de features: {dataset.num_features}")
        print(f"Número de classes: {dataset.num_classes}")
        print("-" * 40)
