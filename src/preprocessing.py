"""Padronização dos atributos do dataset Iris (StandardScaler)."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from eda import load_dataset

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def get_scaled_features(df: pd.DataFrame = None):
    if df is None:
        df = load_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return X_scaled, scaler


def main():
    df = load_dataset()
    X_scaled, _ = get_scaled_features(df)

    print("Antes da padronização:\n", df[FEATURES].describe().loc[["mean", "std"]])

    df_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
    print("\nDepois da padronização:\n", df_scaled.describe().loc[["mean", "std"]])


if __name__ == "__main__":
    main()
