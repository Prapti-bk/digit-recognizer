"""Model architecture. Only knows how to *build* a network, not how to train it."""
from .config import IMAGE_SIZE, NUM_CLASSES


def build_cnn(dropout: float = 0.4):
    """Two conv/pool blocks -> dropout -> dense classifier (compiled, ready to fit)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(dropout),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
