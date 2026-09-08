"""Carregamento e análise exploratória (EDA) do dataset Iris."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

OUTPUT_DIR = "outputs"


def load_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df


def explore(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nTipos:\n", df.dtypes)
    print("\nValores nulos:\n", df.isnull().sum())
    print("\nEstatísticas descritivas:\n", df.describe())
    print("\nDistribuição de espécies:\n", df["species"].value_counts())


def plot_pairplot(df: pd.DataFrame):
    features = df.columns.drop(["target", "species"])
    sns.pairplot(df, vars=features, hue="species")
    plt.savefig(f"{OUTPUT_DIR}/eda_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation(df: pd.DataFrame):
    features = df.columns.drop(["target", "species"])
    corr = df[features].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlação entre atributos")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=150)
    plt.close()


def main():
    df = load_dataset()
    explore(df)
    plot_pairplot(df)
    plot_correlation(df)
    print(f"\nGráficos salvos em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
