import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from digit_recognizer.errors import BlankImageError
from digit_recognizer.preprocessing import crop_to_ink, preprocess, to_model_input


def stroke_image(size=280, invert=False) -> Image.Image:
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).line([(100, 60), (180, 60), (180, 220)], fill=255, width=18)
    return ImageOps.invert(img) if invert else img


def test_output_is_mnist_shaped_and_normalised():
    out = preprocess(stroke_image())
    assert out.shape == (28, 28) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0 and out.max() > 0.5


def test_model_input_has_batch_and_channel_dims():
    assert to_model_input(stroke_image()).shape == (1, 28, 28, 1)


def test_blank_image_raises():
    with pytest.raises(BlankImageError):
        preprocess(Image.new("L", (280, 280), 0))


def test_dark_on_light_gives_same_result_as_light_on_dark():
    assert np.allclose(preprocess(stroke_image()), preprocess(stroke_image(invert=True)), atol=0.05)


def test_centre_of_mass_is_centred():
    out = preprocess(stroke_image())
    rows, cols = np.indices(out.shape)
    assert abs((rows * out).sum() / out.sum() - 13.5) < 1.0
    assert abs((cols * out).sum() / out.sum() - 13.5) < 1.0


def test_crop_to_ink_is_tight():
    pixels = np.zeros((50, 50), dtype=np.uint8)
    pixels[10:20, 30:40] = 255
    assert crop_to_ink(pixels).shape == (10, 10)
