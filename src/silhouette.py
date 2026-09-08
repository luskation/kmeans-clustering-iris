"""Silhouette Score para os mesmos valores de K do Elbow Method."""

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from preprocessing import get_scaled_features
from clustering import run_kmeans_range, K_RANGE

OUTPUT_DIR = "../outputs"


def compute_silhouette_scores(X, results):
    scores = {}
    for k, r in results.items():
        labels = r["model"].labels_
        scores[k] = silhouette_score(X, labels)
    return scores


def plot_silhouette(scores, output_path=f"{OUTPUT_DIR}/silhouette_scores.png"):
    ks = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(7, 5))
    plt.plot(ks, values, marker="o", color="darkorange")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score por K")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    X_scaled, _ = get_scaled_features()
    results = run_kmeans_range(X_scaled, K_RANGE)
    scores = compute_silhouette_scores(X_scaled, results)

    print("K\tSilhouette Score")
    for k, s in scores.items():
        print(f"{k}\t{s:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nMelhor K por Silhouette Score: {best_k} ({scores[best_k]:.4f})")

    plot_silhouette(scores)
    print(f"Gráfico salvo em {OUTPUT_DIR}/silhouette_scores.png")


if __name__ == "__main__":
    main()
