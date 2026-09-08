"""K-Means para múltiplos valores de K e cálculo do WCSS (inertia)."""

from sklearn.cluster import KMeans

from preprocessing import get_scaled_features

K_RANGE = range(2, 11)
RANDOM_STATE = 42


def run_kmeans_range(X, k_range=K_RANGE):
    results = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(X)
        results[k] = {"model": model, "wcss": model.inertia_}
    return results


def main():
    X_scaled, _ = get_scaled_features()
    results = run_kmeans_range(X_scaled)

    print("K\tWCSS")
    for k, r in results.items():
        print(f"{k}\t{r['wcss']:.2f}")


if __name__ == "__main__":
    main()
