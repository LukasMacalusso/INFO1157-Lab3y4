from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch

OUTPUT = Path(__file__).resolve().parent / "output"


def FitIrisKNN(k: int = 3) -> tuple[int, int, list[str], float, np.ndarray, str]:
    iris = cast(Bunch, load_iris())
    names = [str(n) for n in iris.target_names]
    xTrain, xTest, yTrain, yTest = train_test_split(
        iris.data, iris.target, test_size=0.20, random_state=42, stratify=iris.target
    )
    knn = KNeighborsClassifier(n_neighbors=k).fit(xTrain, yTrain)
    yPred = knn.predict(xTest)
    acc = float(accuracy_score(yTest, yPred))
    cm = confusion_matrix(yTest, yPred)
    report = str(classification_report(yTest, yPred, target_names=names))
    return len(xTrain), len(xTest), names, acc, cm, report


def PlotConfusionMatrix(cm: np.ndarray, names: list[str], acc: float) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Clase Predicha")
    ax.set_ylabel("Clase Real")
    ax.set_title(f"Matriz de Confusión KNN (80/20 - Accuracy: {acc * 100:.1f}%)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontweight="bold")
    fig.savefig(OUTPUT / "figura_7_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def RunIrisKNN(k: int = 3) -> None:
    nTrain, nTest, names, acc, cm, report = FitIrisKNN(k)
    PlotConfusionMatrix(cm, names, acc)

    print("\n=== Tarea 7: Iris KNN (80% Train / 20% Test) ===")
    print(f"Train: {nTrain} | Test: {nTest} | k={k} | Accuracy: {acc * 100:.2f}%")
    print(report.strip())
    print()


if __name__ == "__main__":
    RunIrisKNN()
