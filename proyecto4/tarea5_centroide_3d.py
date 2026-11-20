from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = Path(__file__).resolve().parent / "output"


def ComputeCentroid3D() -> tuple[pd.DataFrame, np.ndarray]:
    d3 = np.load(ASSETS / "puntos_3d.npy")
    df = pd.DataFrame(d3)
    df.columns = ["X", "Y", "Z"]
    centroid = df.mean().to_numpy(dtype=float)
    return df, centroid


def PlotCentroid3D(df: pd.DataFrame, centroid: np.ndarray) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4.8))
    for i, (title, showCentroid) in enumerate([("Sin Centroide", False), ("Con Centroide", True)], start=1):
        ax: Any = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(df["X"], df["Y"], df["Z"], c="red", s=30, alpha=0.75)
        if showCentroid:
            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c="blue", s=220, zorder=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=28, azim=-62)
        ax.set_title(title, y=-0.15, fontweight="bold")

    fig.savefig(OUTPUT / "figura_5_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunCentroid3D() -> None:
    df, centroid = ComputeCentroid3D()
    PlotCentroid3D(df, centroid)

    print("\n=== Tarea 5: Centroide 3D ===")
    print(f"DataFrame shape: {df.shape} | Columnas: {list(df.columns)}")
    print(f"Centroide (Azul): X={centroid[0]:.4f}, Y={centroid[1]:.4f}, Z={centroid[2]:.4f}")
    print()


if __name__ == "__main__":
    RunCentroid3D()
