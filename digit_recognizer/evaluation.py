"""Metrics and report plots. Kept separate from training so each can change alone."""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import MnistData

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    classification_report: str


def evaluate(model, data: MnistData, reports_dir: Path) -> EvaluationReport:
    """Score on the test set and save a confusion-matrix image."""
    from sklearn.metrics import classification_report, confusion_matrix

    predicted = np.argmax(model.predict(data.x_test, verbose=0), axis=1)
    accuracy = float((predicted == data.y_test).mean())
    _save_confusion_matrix(confusion_matrix(data.y_test, predicted), reports_dir)
    return EvaluationReport(accuracy, classification_report(data.y_test, predicted, digits=4))


def plot_training_curves(history: dict, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set(title=metric.capitalize(), xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def _save_confusion_matrix(matrix: np.ndarray, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    reports_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(matrix).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    plt.close()
    log.info("Saved confusion matrix to %s", reports_dir)
