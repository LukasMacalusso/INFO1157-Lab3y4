from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = Path(__file__).resolve().parent / "output"

TEST_DATA = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]], dtype=float)
COLORS_RAW = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}
COLORS_FIT = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728"}
NAMES_FIT = {0: "Verde (Sup. Izq.)", 1: "Azul (Inf. Der.)", 2: "Rojo (Sup. Der.)"}


def FitKMeans():
    A = np.load(ASSETS / "A.npy")
    yA = np.load(ASSETS / "y_A.npy")
    model = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(A)
    predictions = model.predict(TEST_DATA)
    return A, yA, model, predictions


def SaveFigure(fig, ax, label: str, filename: str) -> None:
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-11.0, 10.0)
    ax.text(0.04, 0.06, label, transform=ax.transAxes, fontweight="bold")
    fig.savefig(OUTPUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def PlotKMeans(A: np.ndarray, yA: np.ndarray, model: KMeans) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(A[:, 0], A[:, 1], c="#1f77b4", s=20)
    SaveFigure(fig, ax, "Figura 1.1", "figura_1_1.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in np.unique(yA):
        ax.scatter(A[yA == k, 0], A[yA == k, 1], c=COLORS_RAW[int(k)], s=20)
    SaveFigure(fig, ax, "Figura 1.2", "figura_1_2.png")

    centroids = model.cluster_centers_
    labels = np.asarray(model.labels_)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in range(model.n_clusters):
        ax.scatter(A[labels == k, 0], A[labels == k, 1], c=COLORS_FIT[k], s=20)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="*", c="black", s=220, zorder=5)
    SaveFigure(fig, ax, "Figura 1.3", "figura_1_3.png")


def RunKMeans() -> None:
    A, yA, model, predictions = FitKMeans()
    PlotKMeans(A, yA, model)

    print("\n=== Tarea 1: KMeans ===")
    print(f"Etiquetas: {np.unique(yA).tolist()} | Total datos: {len(A)}")
    for idx, c in enumerate(model.cluster_centers_):
        print(f"  Centroide {idx} ({NAMES_FIT[idx]}): [{c[0]:.3f}, {c[1]:.3f}]")
    print("Predicciones test:")
    for point, label in zip(TEST_DATA, predictions):
        print(f"  {point.tolist()} -> Clase {label} ({NAMES_FIT[int(label)]})")
    print()


if __name__ == "__main__":
    RunKMeans()
