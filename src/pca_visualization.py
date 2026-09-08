"""PCA (2D) para visualizar os clusters do K-Means."""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from preprocessing import get_scaled_features

OUTPUT_DIR = "../outputs"
RANDOM_STATE = 42
KS_TO_PLOT = [2, 3]


def plot_clusters_pca(X_pca, labels, k, output_path):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(f"Clusters K-Means (K={k}) — visualização via PCA")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    X_scaled, _ = get_scaled_features()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("Variância explicada por componente:", pca.explained_variance_ratio_)
    print("Variância total explicada (2 componentes): "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    for k in KS_TO_PLOT:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        output_path = f"{OUTPUT_DIR}/pca_clusters_k{k}.png"
        plot_clusters_pca(X_pca, labels, k, output_path)
        print(f"Gráfico salvo em {output_path}")


if __name__ == "__main__":
    main()
