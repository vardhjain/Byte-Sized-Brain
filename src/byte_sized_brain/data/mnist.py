"""MNIST loader for the feed-forward pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ((x_train, y_train), (x_test, y_test)) as float32 784-vectors in [0, 1]."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28 * 28)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28 * 28)
    return (x_train, y_train), (x_test, y_test)


def representative_dataset(x: np.ndarray, n: int = 100) -> Callable[[], Iterator[list[np.ndarray]]]:
    """A real-data representative dataset for static INT8 calibration."""
    n = min(n, len(x))

    def gen() -> Iterator[list[np.ndarray]]:
        for i in range(n):
            yield [x[i : i + 1].astype(np.float32)]

    return gen
