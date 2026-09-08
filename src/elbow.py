"""Elbow Method: gráfico de WCSS x K para apoiar a escolha de K."""

import matplotlib.pyplot as plt

from preprocessing import get_scaled_features
from clustering import run_kmeans_range
from config import K_RANGE, OUTPUT_DIR


def plot_elbow(results, output_path=None):
    output_path = output_path or OUTPUT_DIR / "elbow_method.png"
    ks = list(results.keys())
    wcss = [results[k]["wcss"] for k in ks]

    plt.figure(figsize=(7, 5))
    plt.plot(ks, wcss, marker="o")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_deltas(results):
    ks = list(results.keys())
    wcss = [results[k]["wcss"] for k in ks]
    print("K\tWCSS\t\tQueda (WCSS[k-1]-WCSS[k])")
    print(f"{ks[0]}\t{wcss[0]:.2f}\t\t-")
    for i in range(1, len(ks)):
        delta = wcss[i - 1] - wcss[i]
        print(f"{ks[i]}\t{wcss[i]:.2f}\t\t{delta:.2f}")


def main():
    X_scaled, _ = get_scaled_features()
    results = run_kmeans_range(X_scaled, K_RANGE)
    print_deltas(results)
    plot_elbow(results)
    print(f"\nGráfico salvo em {OUTPUT_DIR / 'elbow_method.png'}")


if __name__ == "__main__":
    main()
