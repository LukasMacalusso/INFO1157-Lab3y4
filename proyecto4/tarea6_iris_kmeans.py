from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

OUTPUT = Path(__file__).resolve().parent / "output"


def FitIrisKMeans() -> tuple[np.ndarray, np.ndarray, float]:
    iris = cast(Bunch, load_iris())
    data = np.asarray(iris.data)
    kmeans = KMeans(n_clusters=3, random_state=20, n_init="auto").fit(data)
    return data, np.asarray(kmeans.labels_), float(kmeans.inertia_ or 0.0)


def PlotIrisKMeans(data: np.ndarray, labels: np.ndarray) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=35)
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    fig.savefig(OUTPUT / "figura_6_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunIrisKMeans() -> None:
    data, labels, inertia = FitIrisKMeans()
    PlotIrisKMeans(data, labels)

    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Tarea 6: Iris KMeans ===")
    print(f"Muestras: {len(data)} | Inercia: {inertia:.4f}")
    print(f"Muestras por cluster: {dict(zip(unique.tolist(), counts.tolist()))}")
    print()


if __name__ == "__main__":
    RunIrisKMeans()
