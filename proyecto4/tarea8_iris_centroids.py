from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

OUTPUT = Path(__file__).resolve().parent / "output"


def FitIrisCentroids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris = cast(Bunch, load_iris())
    data = np.asarray(iris.data)
    kmeans = KMeans(n_clusters=3, random_state=20, n_init="auto").fit(data)
    return data, np.asarray(kmeans.labels_), np.asarray(kmeans.cluster_centers_)


def PlotIrisCentroids(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=35, alpha=0.8)
    ax1.scatter(centroids[:, 0], centroids[:, 1], marker="*", c="red", edgecolors="black", s=260, zorder=10, label="Centroides")
    ax1.set_xlabel("Sepal Length (cm)")
    ax1.set_ylabel("Sepal Width (cm)")
    ax1.set_title("Clusters y Centroides (Sépalos)")
    ax1.legend()

    ax2.scatter(data[:, 2], data[:, 3], c=labels, cmap="viridis", s=35, alpha=0.8)
    ax2.scatter(centroids[:, 2], centroids[:, 3], marker="*", c="red", edgecolors="black", s=260, zorder=10, label="Centroides")
    ax2.set_xlabel("Petal Length (cm)")
    ax2.set_ylabel("Petal Width (cm)")
    ax2.set_title("Clusters y Centroides (Pétalos)")
    ax2.legend()

    fig.savefig(OUTPUT / "figura_8_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunIrisCentroids() -> None:
    data, labels, centroids = FitIrisCentroids()
    PlotIrisCentroids(data, labels, centroids)

    print("\n=== Tarea 8: Iris KMeans + Centroides ===")
    for idx, c in enumerate(centroids):
        print(f"  Centroide {idx}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, {c[3]:.4f}]")
    print()


if __name__ == "__main__":
    RunIrisCentroids()
