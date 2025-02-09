# Comparação de Redes GNNs para Classificação de Grafos

Este projeto implementa e compara 4 arquiteturas de redes neurais baseadas em grafos (GNNs) para a tarefa de **classificação de nós** nos datasets **Cora**, **Citeseer** e **PubMed**.

📌 Descrição dos Datasets
Os três datasets são provenientes do benchmark de aprendizado em grafos Planetoid e são muito utilizados para tarefas de classificação de nós. Todos representam redes de citações acadêmicas, onde:

Os nós representam artigos científicos.
As arestas representam citações entre artigos.
Os atributos dos nós representam informações sobre os artigos (exemplo: palavras-chave em uma representação de bag-of-words).
As classes representam a categoria temática do artigo.

1️⃣ Cora
📊 Nós: 2.708 
🔗 Arestas: 5.429
🔤 Atributos por nó: 1.433
🎯 Classes: 7 
            Redes Neurais 🧠
            Aprendizado de Máquina 🤖
            Recuperação de Informação 🔎
            Visão Computacional 👁️
            Processamento de Linguagem Natural 🗣️
            Sistemas Operacionais 🖥️
            Teoria da Computação 📏
📌 Descrição: O dataset Cora contém artigos acadêmicos classificados em 7 categorias diferentes. Os nós representam artigos, e as arestas representam citações entre os artigos.

2️⃣ Citeseer
📊 Nós: 3.327
🔗 Arestas: 4.732
🔤 Atributos por nó: 3.703
🎯 Classes: 6 (categorias de pesquisa, como IA, Banco de Dados, etc.)
            Agentes 🤖
            Aprendizado de Máquina 📊
            Redes Neurais 🧠
            Banco de Dados 💾
            Sistemas Operacionais 🖥️
            Teoria 📏
📌 Descrição: O Citeseer é um dataset mais difícil do que o Cora porque tem menos conexões por nó e um maior número de atributos. Também representa um grafo de citações acadêmicas.

3️⃣ PubMed
📊 Nós: 19.717
🔗 Arestas: 44.324
🔤 Atributos por nó: 500
🎯 Classes: 3 (tipos de doenças relacionadas ao diabetes)
            Diabetes Tipo 1 🏥
            Diabetes Tipo 2 🩸
            Diabetes com complicações secundárias ⚠️
📌 Descrição: O dataset PubMed contém artigos médicos relacionados a diabetes classificados em 3 categorias. É significativamente maior que Cora e Citeseer.

As arquiteturas testadas são:

- **GCN (Graph Convolutional Network)**

# 📌 Graph Convolutional Network (GCN)

## 🔍 O que é o GCN?
O **Graph Convolutional Network (GCN)** é uma rede neural projetada para trabalhar com **grafos**, permitindo que os nós aprendam representações baseadas na estrutura do grafo. Foi introduzido por **Thomas Kipf e Max Welling** em 2017 e é amplamente utilizado para tarefas como **classificação de nós, predição de links e aprendizado de representações em grafos**.

---

## 🚀 Como o GCN funciona?
A ideia central do GCN é permitir que cada nó combine suas **próprias features** com as dos seus **vizinhos**, extraindo informações estruturais ao longo de várias camadas da rede.

A atualização dos nós é definida pela equação:
$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$


Dessa forma, a cada camada, os nós acumulam informações de seus vizinhos de forma iterativa.

---

## 🔗 Fluxo de um GCN
1. **Entrada:**
   - Um grafo representado pela matriz de adjacência \( A \);
   - Uma matriz de features \( X \), onde cada linha corresponde a um nó.
2. **Propagação:**
   - Cada nó agrega informações dos seus vizinhos.
3. **Saída:**
   - Embeddings finais dos nós, que podem ser usadas para classificação ou outras tarefas.

---

## 📌 Vantagens do GCN
✔️ Captura **estruturas** e **relações** do grafo.  
✔️ É eficiente devido à **aproximação espectral**.  
✔️ Funciona bem com **dados esparsos**.  

## ⚠️ Limitações do GCN
❌ Sofre com **oversmoothing** (as representações dos nós se tornam muito semelhantes em camadas profundas).  
❌ Tem dificuldade em capturar **conectividades complexas** (exemplo: redes sociais densas).  

---

## 🛠 Implementação no Projeto
No seu projeto, o GCN está implementado na classe `GCN` dentro do arquivo `modelo.py`. Ele recebe:
- O **número de features** dos nós (entrada);
- A **dimensão oculta** para aprendizado intermedário;
- O **número de classes** para classificação final.

O modelo realiza **convoluções sobre o grafo**, propagando informações entre nós vizinhos para gerar predições mais precisas.

Se quiser explorar outras arquiteturas (GAT, SGC, MPNN), basta conferir os respectivos arquivos no projeto! 🚀



- **GAT (Graph Attention Network)**
- **SGC (Simplifying Graph Convolution)**
- **MPNN (Message Passing Neural Network)**

Os experimentos são conduzidos para avaliar o desempenho dessas arquiteturas em termos de acurácia.

---

## 📁 Estrutura do Projeto

```
📂 PPGCC_TGRAF
│── 📂 utils
|   |── __init__.py    
│   ├── data_loader.py  # Carrega os datasets
│── 📂 modelo
│   │── __init__.py     # arquivo com configuração dos datasets
│   ├── gcn.py          # Implementação do GCN
│   ├── gat.py          # Implementação do GAT
│   ├── sgc.py          # Implementação do SGC
│   ├── mpnn.py         # Implementação do MPNN
│── train.py            # Treina um único modelo
│── main.py             # Executa a comparação entre os modelos
│── requirements.txt     # Dependências do projeto
│── README.md            # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto

### 1️⃣ Instalar Dependências
Certifique-se de ter o **Python 3.11** instalado. Em seguida, instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

### 2️⃣ Baixar os Datasets
Antes de treinar os modelos, baixe os datasets necessários executando o script:

```bash
python utils/data_loader.py
```

Isso baixará os datasets **Cora**, **Citeseer** e **PubMed** automaticamente na pasta `data/`.

### 3️⃣ Testar o Treinamento de um Modelo
Para testar se um modelo específico está funcionando corretamente, execute:

```bash
python train.py
```

Isso treina um único modelo definido dentro do `train.py` e exibe a evolução da perda e acurácia.

### 4️⃣ Executar a Comparação Completa
Agora, para treinar **todos os modelos em todos os datasets** e comparar os resultados, rode:

```bash
python main.py
```

Isso:
✅ Carrega os datasets
✅ Treina cada modelo
✅ Avalia os resultados
✅ Salva a comparação em um arquivo CSV (`comparacao_resultados.csv`)

---

## 📊 Sobre os Datasets
Os datasets utilizados são amplamente usados em benchmarks para **aprendizado de grafos**:

- **Cora**: Contém artigos científicos como nós e citações entre eles como arestas. Cada nó possui uma representação baseada em palavras-chave e uma classe de categoria.
- **Citeseer**: Semelhante ao Cora, mas com um conjunto de artigos de diferentes áreas e maior dificuldade na classificação.
- **PubMed**: Artigos da área biomédica com informações estruturadas para tarefas de classificação.

Todos os datasets são carregados usando a biblioteca **torch_geometric.datasets.Planetoid**.

---

## 📌 Objetivo do Projeto
O objetivo é comparar o desempenho de diferentes **redes neurais baseadas em grafos** na tarefa de classificação de nós. A métrica principal utilizada é a **acurácia** nos datasets testados.

No final da execução do `main.py`, será gerado um CSV com os resultados para facilitar a análise da performance de cada modelo.

---

## 📩 Contato
Caso tenha dúvidas ou sugestões, sinta-se à vontade para entrar em contato! 🚀

