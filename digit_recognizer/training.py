"""Training workflow: load data -> build -> fit -> evaluate -> save."""
import logging
from dataclasses import dataclass

from .config import TrainingConfig
from .data import load_mnist
from .evaluation import EvaluationReport, evaluate, plot_training_curves
from .model import build_cnn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    report: EvaluationReport
    history: dict


def train(config: TrainingConfig | None = None) -> TrainingResult:
    """Run one full training job described entirely by `config`."""
    from tensorflow import keras

    config = config or TrainingConfig()

    keras.utils.set_random_seed(config.seed)          # reproducible runs
    data = load_mnist()
    log.info("Loaded MNIST: %d train / %d test images", len(data.x_train), len(data.x_test))

    model = build_cnn(config.dropout)
    history = model.fit(
        data.x_train, data.y_train,
        epochs=config.epochs, batch_size=config.batch_size,
        validation_split=config.validation_split, verbose=2,
    ).history

    report = evaluate(model, data, config.reports_dir)
    plot_training_curves(history, config.reports_dir)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    log.info("Test accuracy %.4f. Model saved to %s", report.accuracy, config.model_path)
    return TrainingResult(report, history)
