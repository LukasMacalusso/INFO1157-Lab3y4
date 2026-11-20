from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = Path(__file__).resolve().parent / "output"

TEST_DATA = np.array([[-7, -6], [1.5, -6.5], [7.9, 0.5], [5.5, 10]], dtype=float)
COLORS_RAW = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4", 3: "#bcbd22"}
COLORS_FIT = {0: "#bcbd22", 1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
NAMES_FIT = {0: "Amarillo (Der.)", 1: "Azul (Inf. Centro)", 2: "Verde (Sup.)", 3: "Rojo (Inf. Izq.)"}


def FitMeanShift():
    X = np.load(ASSETS / "X.npy")
    yX = np.load(ASSETS / "y_X.npy")
    bandwidth = float(estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=42))
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    predictions = model.predict(TEST_DATA)
    return X, yX, model, predictions


def SaveFigure(fig, ax, label: str, filename: str) -> None:
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    ax.set_xlim(-10.5, 10.8)
    ax.set_ylim(-12.0, 13.0)
    ax.text(0.72, 0.08, label, transform=ax.transAxes, fontweight="bold")
    fig.savefig(OUTPUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def PlotMeanShift(X: np.ndarray, yX: np.ndarray, model: MeanShift) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(X[:, 0], X[:, 1], c="#1f77b4", s=20)
    SaveFigure(fig, ax, "Figura 2.1", "figura_2_1.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in np.unique(yX):
        ax.scatter(X[yX == k, 0], X[yX == k, 1], c=COLORS_RAW[int(k)], s=20)
    SaveFigure(fig, ax, "Figura 2.2", "figura_2_2.png")

    centroids = model.cluster_centers_
    labels = np.asarray(model.labels_)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in range(len(centroids)):
        ax.scatter(X[labels == k, 0], X[labels == k, 1], c=COLORS_FIT.get(k, "gray"), s=20)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="*", c="black", s=220, zorder=5)
    SaveFigure(fig, ax, "Figura 2.3", "figura_2_3.png")


def RunMeanShift() -> None:
    X, yX, model, predictions = FitMeanShift()
    PlotMeanShift(X, yX, model)

    centroids = model.cluster_centers_
    print("\n=== Tarea 2: MeanShift ===")
    print(f"Etiquetas: {np.unique(yX).tolist()} | Total datos: {len(X)} | Clusters: {len(centroids)}")
    for idx, c in enumerate(centroids):
        print(f"  Centroide {idx} ({NAMES_FIT.get(idx, '')}): [{c[0]:.3f}, {c[1]:.3f}]")
    print("Predicciones test:")
    for point, label in zip(TEST_DATA, predictions):
        print(f"  {point.tolist()} -> Clase {label} ({NAMES_FIT.get(int(label), '')})")
    print()


if __name__ == "__main__":
    RunMeanShift()
