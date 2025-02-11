import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dados do experimento
data = {
    "Dataset": ["Cora", "Cora", "Cora", "Cora", "Citeseer", "Citeseer", "Citeseer", "Citeseer", "PubMed", "PubMed", "PubMed", "PubMed"],
    "Modelo": ["GCN", "GAT", "SGC", "MPNN", "GCN", "GAT", "SGC", "MPNN", "GCN", "GAT", "SGC", "MPNN"],
    "Acurácia": [0.806, 0.739, 0.799, 0.746, 0.681, 0.670, 0.663, 0.575, 0.791, 0.727, 0.781, 0.701],
    "Tempo (s)": [2.689, 2.870, 19.005, 5.659, 5.667, 6.561, 56.356, 16.389, 6.985, 10.176, 56.237, 22.110]
}

df = pd.DataFrame(data)

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")

# Gráfico de Acurácia
plt.figure(figsize=(10, 6))
sns.barplot(x="Dataset", y="Acurácia", hue="Modelo", data=df, palette="viridis")
plt.title("Comparação de Acurácia dos Modelos")
plt.ylim(0.5, 0.85)  # Define um limite para melhor visualização
plt.legend(title="Modelo")
plt.savefig("comparacao_acuracia.png")  # Salvar a figura
plt.show()

# Gráfico de Tempo de Execução
plt.figure(figsize=(10, 6))
sns.barplot(x="Dataset", y="Tempo (s)", hue="Modelo", data=df, palette="magma")
plt.title("Tempo de Treinamento dos Modelos")
plt.ylim(0, 60)  # Define um limite para melhor visualização
plt.legend(title="Modelo")
plt.savefig("comparacao_tempo.png")  # Salvar a figura
plt.show()
