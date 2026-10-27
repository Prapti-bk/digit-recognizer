"""Dataset loading. TensorFlow is imported lazily to keep package import cheap."""
from dataclasses import dataclass

import numpy as np

from .config import IMAGE_SIZE


@dataclass(frozen=True)
class MnistData:
    """Train/test split, images shaped (N, 28, 28, 1) in [0, 1]; labels are ints 0-9."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def normalize(images: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> float32 (N, 28, 28, 1) scaled to [0, 1]."""
    return (images.astype("float32") / 255.0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def load_mnist() -> MnistData:
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return MnistData(normalize(x_train), y_train, normalize(x_test), y_test)
