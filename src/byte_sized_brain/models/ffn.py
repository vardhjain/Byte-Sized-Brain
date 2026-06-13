"""Feed-forward network for MNIST."""

from __future__ import annotations


def build_ffn(input_dim: int = 784, hidden: int = 128, num_classes: int = 10):
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ffn_mnist",
    )
