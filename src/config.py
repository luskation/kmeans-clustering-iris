"""Constantes compartilhadas pelo pipeline de clustering."""

from pathlib import Path

RANDOM_STATE = 42
K_RANGE = range(2, 11)

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
