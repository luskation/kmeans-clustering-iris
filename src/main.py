"""Executa o pipeline completo de clustering K-Means na base Iris, do EDA à
comparação final com os rótulos reais de espécie."""

import eda
import preprocessing
import clustering
import elbow
import silhouette
import pca_visualization
import compare_clusters


def run_pipeline():
    steps = [
        ("Análise exploratória (EDA)", eda.main),
        ("Padronização (StandardScaler)", preprocessing.main),
        ("K-Means + WCSS (K=2..10)", clustering.main),
        ("Elbow Method", elbow.main),
        ("Silhouette Score", silhouette.main),
        ("PCA + visualização dos clusters", pca_visualization.main),
        ("Comparação clusters vs espécies reais", compare_clusters.main),
    ]

    for title, step in steps:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        step()


if __name__ == "__main__":
    run_pipeline()
