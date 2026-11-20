from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent / "output"

VECTORS = {
    "A": np.array([2, 1, 0, 2, 0, 1, 1, 1], dtype=float),
    "B": np.array([2, 1, 1, 1, 1, 0, 1, 1], dtype=float),
    "P": np.array([1, 2, 3, 0, 4, 6, 7, 9], dtype=float),
    "Q": np.array([2, 4, 3, 1, 8, 2, 4, 1], dtype=float),
    "S": np.array([2, 1, 4, 7, 1, 4, 5, 6], dtype=float),
    "T": np.array([3, 3, 3, 6, 1, 1, 7, 8], dtype=float),
}
PAIRS = [("A", "B"), ("P", "Q"), ("S", "T")]


def CosineMetrics(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    cosSim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    cosDist = 1.0 - cosSim
    rad = float(np.arccos(np.clip(cosSim, -1.0, 1.0)))
    deg = float(np.degrees(rad))
    return cosSim, cosDist, rad, deg


def ComputeAllCosine() -> list[tuple[str, float, float, float, float]]:
    return [
        (f"({k1}, {k2})", *CosineMetrics(VECTORS[k1], VECTORS[k2]))
        for k1, k2 in PAIRS
    ]


def PlotCosineDistance(results: list[tuple[str, float, float, float, float]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = [r[0] for r in results]
    sims = [r[1] for r in results]
    dists = [r[2] for r in results]
    degs = [r[4] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    x = np.arange(len(names))
    ax1.bar(x - 0.18, sims, 0.35, label="Similaridad Cos(θ)", color="#2ca02c")
    ax1.bar(x + 0.18, dists, 0.35, label="Distancia (1 - Cos)", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Similaridad vs Distancia del Coseno")
    ax1.legend()

    ax2.bar(names, degs, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.45)
    ax2.set_ylabel("Grados (°)")
    ax2.set_title("Ángulo θ entre Vectores")

    fig.savefig(OUTPUT / "figura_4_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunCosineDistance() -> None:
    results = ComputeAllCosine()
    PlotCosineDistance(results)

    print("\n=== Tarea 4: Distancia y Similaridad del Coseno ===")
    for name, sim, dist, rad, deg in results:
        print(f"  {name}: Cos(θ) = {sim:.4f} | Dist = {dist:.4f} | θ = {rad:.4f} rad ({deg:.2f}°)")
    print("  θ = 0 rad   -> Cos = 1, Dist = 0 (misma dirección, máxima similaridad)")
    print("  θ = π/2 rad -> Cos = 0, Dist = 1 (vectores ortogonales, sin similaridad)")
    print()


if __name__ == "__main__":
    RunCosineDistance()
