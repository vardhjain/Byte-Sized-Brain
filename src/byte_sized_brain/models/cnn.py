"""MobileNetV2 backbone + classification head for CIFAR-10."""

from __future__ import annotations


def build_mobilenet(img_size: int = 96, num_classes: int = 10):
    """Return ``(model, base)``; ``base`` is exposed so the pipeline can unfreeze it."""
    from tensorflow import keras

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    model = keras.Sequential(
        [
            base,
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_cifar10",
    )
    return model, base
