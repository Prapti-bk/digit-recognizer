"""Turn an arbitrary hand-drawn image into an MNIST-style model input.

Why this exists: MNIST digits are cropped, scaled so the longest side is 20px,
and centred by centre-of-mass in a 28x28 frame. A raw canvas drawing differs
from that, which hurts accuracy. Replicating the recipe fixes it.

Pure NumPy + Pillow (no TensorFlow). Each step is a small function so it can be
tested and changed in isolation; `preprocess` simply chains them.
"""
import numpy as np
from PIL import Image

from .config import DIGIT_BOX, IMAGE_SIZE
from .errors import BlankImageError

INK_THRESHOLD = 30             # pixel values above this count as "ink"
LIGHT_BACKGROUND_MEAN = 127    # mean brightness above this => dark ink on light paper


def to_white_on_black(image: Image.Image) -> np.ndarray:
    """Greyscale uint8 array with a white digit on a black background."""
    pixels = np.array(image.convert("L"))
    if pixels.mean() > LIGHT_BACKGROUND_MEAN:
        pixels = 255 - pixels
    return pixels


def crop_to_ink(pixels: np.ndarray) -> np.ndarray:
    """Crop to the tight bounding box of the ink. Raises BlankImageError if none."""
    rows, cols = np.where(pixels > INK_THRESHOLD)
    if rows.size == 0:
        raise BlankImageError("The image contains no digit to recognise.")
    return pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def scale_to_box(pixels: np.ndarray, box: int = DIGIT_BOX) -> Image.Image:
    """Scale so the longest side equals `box`, preserving the aspect ratio."""
    height, width = pixels.shape
    ratio = box / max(height, width)
    new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return Image.fromarray(pixels).resize(new_size, Image.LANCZOS)


def paste_centered(digit: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Place the digit in the middle of a black `size` x `size` frame."""
    frame = Image.new("L", (size, size), 0)
    frame.paste(digit, ((size - digit.width) // 2, (size - digit.height) // 2))
    return np.array(frame, dtype=np.float32)


def recenter_by_mass(frame: np.ndarray) -> np.ndarray:
    """Shift the image so its centre of mass sits at the frame centre."""
    centre = (frame.shape[0] - 1) / 2
    row_idx, col_idx = np.indices(frame.shape)
    total = frame.sum()
    com_row = (row_idx * frame).sum() / total
    com_col = (col_idx * frame).sum() / total
    return np.roll(frame, (round(centre - com_row), round(centre - com_col)), axis=(0, 1))


def preprocess(image: Image.Image) -> np.ndarray:
    """Full pipeline -> float32 array of shape (28, 28) with values in [0, 1]."""
    pixels = to_white_on_black(image)
    pixels = crop_to_ink(pixels)
    framed = paste_centered(scale_to_box(pixels))
    return recenter_by_mass(framed) / 255.0


def to_model_input(image: Image.Image) -> np.ndarray:
    """Pipeline output reshaped to the CNN's input: (1, 28, 28, 1)."""
    return preprocess(image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
