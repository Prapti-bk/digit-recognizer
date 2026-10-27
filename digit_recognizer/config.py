"""Single source of truth for every tunable value and constant."""
from dataclasses import dataclass
from pathlib import Path

IMAGE_SIZE = 28          # MNIST images are 28x28 pixels
DIGIT_BOX = 20           # MNIST digits occupy a 20x20 box inside that frame
NUM_CLASSES = 10
DEFAULT_MODEL_PATH = Path("artifacts/mnist_cnn.keras")


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters and output locations for one training run (immutable)."""

    epochs: int = 8
    batch_size: int = 128
    validation_split: float = 0.1
    dropout: float = 0.4
    seed: int = 42
    model_path: Path = DEFAULT_MODEL_PATH
    reports_dir: Path = Path("artifacts/reports")
