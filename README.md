# K-Means Clustering — Iris

TP02 da disciplina GCC 128 (Inteligência Artificial) — UFLA.

Clustering não supervisionado (K-Means) na base Iris. Os rótulos de espécie são usados
apenas para análise final, nunca como entrada do modelo.

## Pipeline
1. Carregamento e exploração do dataset (EDA)
2. Padronização dos atributos (StandardScaler)
3. K-Means para K = 2 a 10
4. Elbow Method (WCSS)
5. Silhouette Score
6. PCA para visualização dos clusters
7. Comparação clusters vs espécies reais

## Setup
```bash
pip install -r requirements.txt
```

## Estrutura
- `src/` — código do pipeline
- `outputs/` — gráficos gerados
