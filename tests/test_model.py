"""Smoke tests that need TensorFlow (skipped automatically when it isn't installed)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("tensorflow")

from digit_recognizer.model import build_cnn
from digit_recognizer.predictor import DigitPredictor


def test_model_output_shape():
    assert build_cnn().output_shape == (None, 10)


def test_save_load_predict_roundtrip(tmp_path):
    model = build_cnn()
    model.fit(np.random.rand(32, 28, 28, 1), np.random.randint(0, 10, 32), epochs=1, verbose=0)
    path = tmp_path / "m.keras"
    model.save(path)

    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).line([(20, 20), (80, 80)], fill=255, width=8)
    result = DigitPredictor.from_path(path).predict(img)
    assert 0 <= result.digit <= 9 and 0 < result.confidence <= 1
