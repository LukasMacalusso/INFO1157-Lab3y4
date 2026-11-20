from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

OUTPUT = Path(__file__).resolve().parent / "output"

HEIGHT_TRAIN = np.array([1.60, 1.65, 1.70, 1.73, 1.80])
WEIGHT_TRAIN = np.array([60.0, 65.0, 72.3, 75.0, 80.0])
HEIGHT_TEST = np.array([1.58, 1.62, 1.69, 1.76, 1.82])


def FitLinearRegression():
    xTrain = HEIGHT_TRAIN.reshape(-1, 1)
    model = linear_model.LinearRegression().fit(xTrain, WEIGHT_TRAIN)
    yTrainPred = model.predict(xTrain)
    rss = float(np.sum((WEIGHT_TRAIN - yTrainPred) ** 2))
    r2 = float(model.score(xTrain, WEIGHT_TRAIN))
    yTestPred = model.predict(HEIGHT_TEST.reshape(-1, 1))
    return model, yTestPred, rss, r2


def PlotLinearRegression(model: linear_model.LinearRegression, yTestPred: np.ndarray, rss: float) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(HEIGHT_TRAIN, WEIGHT_TRAIN, color="royalblue", s=70, label="Entrenamiento")
    ax.scatter(HEIGHT_TEST, yTestPred, color="darkorange", marker="s", s=65, label="Predicciones Test")
    xLine = np.linspace(1.55, 1.85, 100).reshape(-1, 1)
    ax.plot(xLine, model.predict(xLine), color="crimson", lw=2, label="Regresión Lineal")
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("Peso (Kg)")
    ax.set_title(f"Peso vs Altura (RSS = {rss:.4f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.savefig(OUTPUT / "figura_3_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunLinearRegression() -> None:
    model, yTestPred, rss, r2 = FitLinearRegression()
    PlotLinearRegression(model, yTestPred, rss)

    print("\n=== Tarea 3: Regresión Lineal ===")
    print(f"Peso = {model.coef_[0]:.4f} * Altura + ({model.intercept_:.4f})")
    print(f"RSS = {rss:.4f} kg² | R² = {r2:.4f}")
    for h, w in zip(HEIGHT_TEST, yTestPred):
        print(f"  Altura {h:.2f} m -> Peso predicho: {w:.2f} Kg")
    print()


if __name__ == "__main__":
    RunLinearRegression()
