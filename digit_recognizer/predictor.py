"""Inference service: the only thing the UI and CLI need to know about.

`DigitPredictor` depends on a tiny `ModelLike` protocol rather than on Keras,
so it can be unit-tested with a fake model and the real backend can be swapped
(e.g. ONNX, TFLite) without touching callers.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .errors import ModelNotFoundError
from .preprocessing import preprocess, to_model_input


class ModelLike(Protocol):
    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Return class probabilities with shape (batch, 10)."""


@dataclass(frozen=True)
class Prediction:
    digit: int
    confidence: float
    probabilities: tuple[float, ...]
    model_input: np.ndarray = field(compare=False, repr=False)   # (28, 28) what the CNN saw


class DigitPredictor:
    def __init__(self, model: ModelLike) -> None:
        self._model = model

    @classmethod
    def from_path(cls, path: Path) -> "DigitPredictor":
        if not Path(path).exists():
            raise ModelNotFoundError(
                f"No model at '{path}'. Train one first: python -m digit_recognizer train"
            )
        from tensorflow import keras
        return cls(keras.models.load_model(path))

    def predict(self, image: Image.Image) -> Prediction:
        """Raises BlankImageError if the image has no digit in it."""
        probs = np.asarray(self._model.predict(to_model_input(image), verbose=0))[0]
        digit = int(probs.argmax())
        return Prediction(digit, float(probs[digit]), tuple(map(float, probs)), preprocess(image))
