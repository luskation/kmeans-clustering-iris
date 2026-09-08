"""Comparação dos clusters do K-Means com os rótulos reais de espécie."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from eda import load_dataset
from preprocessing import get_scaled_features

OUTPUT_DIR = "../outputs"
RANDOM_STATE = 42
KS_TO_COMPARE = [2, 3]


def crosstab_clusters_species(labels, species):
    return pd.crosstab(species, labels, rownames=["Espécie real"], colnames=["Cluster"])


def plot_crosstab(ct, k, output_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Espécie real x Cluster (K={k})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    df = load_dataset()
    X_scaled, _ = get_scaled_features(df)
    species = df["species"]

    for k in KS_TO_COMPARE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)

        ct = crosstab_clusters_species(labels, species)
        print(f"\n=== K={k} ===")
        print(ct)

        ari = adjusted_rand_score(df["target"], labels)
        nmi = normalized_mutual_info_score(df["target"], labels)
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")

        output_path = f"{OUTPUT_DIR}/crosstab_k{k}.png"
        plot_crosstab(ct, k, output_path)
        print(f"Gráfico salvo em {output_path}")


if __name__ == "__main__":
    main()
