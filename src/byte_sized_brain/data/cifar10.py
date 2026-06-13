"""CIFAR-10 loader for the MobileNetV2 pipeline (resize + MobileNet preprocess)."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def load_cifar10(
    img_size: int = 96,
    *,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return resized, MobileNetV2-preprocessed CIFAR-10 (floats in [-1, 1])."""
    import tensorflow as tf
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    if train_subset:
        x_train, y_train = x_train[:train_subset], y_train[:train_subset]
    if test_subset:
        x_test, y_test = x_test[:test_subset], y_test[:test_subset]

    preprocess = keras.applications.mobilenet_v2.preprocess_input

    def prep(x: np.ndarray) -> np.ndarray:
        x = tf.image.resize(x.astype("float32"), (img_size, img_size)).numpy()
        return preprocess(x)

    return (prep(x_train), y_train), (prep(x_test), y_test)


def representative_dataset(x: np.ndarray, n: int = 100) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Real CIFAR images for static INT8 calibration."""
    n = min(n, len(x))

    def gen() -> Iterator[list[np.ndarray]]:
        for i in range(n):
            yield [x[i : i + 1].astype(np.float32)]

    return gen
