import numpy as np
import pytest
from PIL import Image, ImageDraw

from digit_recognizer.errors import BlankImageError, ModelNotFoundError
from digit_recognizer.predictor import DigitPredictor


class FakeModel:
    """Always answers '7' with 90% confidence, and records what it was given."""

    def __init__(self):
        self.last_input = None

    def predict(self, x, verbose=0):
        self.last_input = x
        probs = np.full((len(x), 10), 0.1 / 9)
        probs[:, 7] = 0.9
        return probs


def drawn_digit() -> Image.Image:
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).line([(20, 20), (80, 20), (50, 90)], fill=255, width=8)
    return img


def test_prediction_fields():
    model = FakeModel()
    result = DigitPredictor(model).predict(drawn_digit())
    assert result.digit == 7
    assert result.confidence == pytest.approx(0.9)
    assert sum(result.probabilities) == pytest.approx(1.0)
    assert model.last_input.shape == (1, 28, 28, 1)


def test_blank_image_propagates_domain_error():
    with pytest.raises(BlankImageError):
        DigitPredictor(FakeModel()).predict(Image.new("L", (50, 50), 0))


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train"):
        DigitPredictor.from_path(tmp_path / "nope.keras")
