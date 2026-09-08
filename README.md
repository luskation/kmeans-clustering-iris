# K-Means Clustering — Iris Dataset

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20PCA-orange)
![status](https://img.shields.io/badge/status-completo-brightgreen)

Pipeline completo de **clustering não supervisionado** aplicado ao dataset Iris:
padronização → K-Means (K=2 a 10) → Elbow Method → Silhouette Score → PCA →
comparação com as espécies reais. Os rótulos de espécie **nunca entram no modelo**
— são usados só na etapa final, para validar o que o algoritmo encontrou sozinho.

> Trabalho da disciplina GCC128 (Inteligência Artificial) — UFLA.

## Achado principal

O projeto não escolhe K só pelo gráfico do cotovelo — ele confronta três critérios
diferentes, e eles **discordam entre si**, o que é o resultado mais interessante da
análise:

| Critério | K sugerido | Por quê |
|---|---|---|
| Elbow Method (WCSS) | 3 | maior queda de WCSS acontece de K=2→3 (−82.5) |
| Silhouette Score | 2 | 0.582 em K=2 vs. 0.460 em K=3 — clusters mais "compactos" |
| Adjusted Rand Index (vs. espécies reais) | **3** | 0.620 em K=3 vs. 0.568 em K=2 |

**Por que o Silhouette prefere K=2 mesmo havendo 3 espécies reais:** *versicolor* e
*virginica* se sobrepõem parcialmente no espaço dos 4 atributos, então o K-Means com
K=2 as funde num único cluster bem separado da *setosa* — geometricamente compacto,
mas biologicamente incompleto. Com K=3 o modelo separa as duas espécies com 75/100
acertos, refletindo com mais fidelidade a estrutura real, ainda que com menor
Silhouette Score.

Conclusão prática: **métrica interna de coesão (Silhouette) e validação externa
contra rótulos reais (ARI) não são a mesma coisa** — e só dá pra saber isso porque o
dataset tem rótulos conhecidos, o que normalmente não é o caso em clustering real.

## Pipeline

| Etapa | Script | Saída |
|---|---|---|
| 1. EDA | `src/eda.py` | `outputs/eda_pairplot.png`, `outputs/eda_correlation.png` |
| 2. Padronização | `src/preprocessing.py` | médias ≈ 0, desvio-padrão ≈ 1 |
| 3. K-Means + WCSS | `src/clustering.py` | WCSS para K=2..10 |
| 4. Elbow Method | `src/elbow.py` | `outputs/elbow_method.png` |
| 5. Silhouette Score | `src/silhouette.py` | `outputs/silhouette_scores.png` |
| 6. PCA (2D) | `src/pca_visualization.py` | `outputs/pca_clusters_k{2,3}.png` |
| 7. Comparação com espécies reais | `src/compare_clusters.py` | `outputs/crosstab_k{2,3}.png`, ARI, NMI |

Todas as constantes compartilhadas (seed, range de K, atributos, diretório de saída)
ficam centralizadas em `src/config.py`.

## Resultados

**PCA explica 95.8% da variância** dos 4 atributos originais em apenas 2 componentes
— o que torna a visualização 2D uma representação fiel dos dados, não uma
simplificação grosseira.

**Matriz espécie real × cluster (K=3):**

| Espécie \ Cluster | 0 | 1 | 2 |
|---|---|---|---|
| setosa | 0 | 50 | 0 |
| versicolor | 39 | 0 | 11 |
| virginica | 14 | 0 | 36 |

*Setosa* é separada com 100% de pureza em qualquer K — é linearmente separável das
outras duas desde o espaço original de atributos, o que já aparecia no heatmap de
correlação da EDA.

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python3 main.py          # roda o pipeline completo, etapa por etapa
```

Ou rodar uma etapa isolada, ex. `python3 elbow.py`.

## Stack

`numpy` · `pandas` · `scikit-learn` (KMeans, PCA, StandardScaler, métricas de
clustering) · `matplotlib` / `seaborn` para visualização.

## Estrutura

```
kmeans/
├── src/
│   ├── config.py              # constantes compartilhadas
│   ├── eda.py                 # carregamento + EDA
│   ├── preprocessing.py       # StandardScaler
│   ├── clustering.py          # K-Means para K=2..10 + WCSS
│   ├── elbow.py                # Elbow Method
│   ├── silhouette.py          # Silhouette Score
│   ├── pca_visualization.py   # PCA 2D + scatter dos clusters
│   ├── compare_clusters.py    # crosstab, ARI, NMI vs. espécies reais
│   └── main.py                 # orquestra o pipeline completo
├── outputs/                    # gráficos gerados (.png)
└── requirements.txt
```
