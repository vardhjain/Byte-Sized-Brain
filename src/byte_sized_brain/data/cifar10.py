"""CIFAR-10 loader for the MobileNetV2 pipeline (resize + MobileNet preprocess).

Resizing to 96x96 float32 costs about 110 KB per image, so the full 50k training
split is roughly 5.5 GB. :func:`load_split` therefore preprocesses only the split
(and the number of images) a stage actually needs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from . import representative_dataset

Split = Literal["train", "test"]


def _prep(
    x: np.ndarray, y: np.ndarray, img_size: int, subset: int | None
) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf
    from tensorflow import keras

    if subset:
        x, y = x[:subset], y[:subset]
    x = tf.image.resize(x.astype("float32"), (img_size, img_size)).numpy()
    return keras.applications.mobilenet_v2.preprocess_input(x), y.reshape(-1)


def load_split(
    split: Split, img_size: int = 96, subset: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One resized, MobileNetV2-preprocessed split (floats in [-1, 1])."""
    from tensorflow import keras

    train, test = keras.datasets.cifar10.load_data()
    x, y = train if split == "train" else test
    return _prep(x, y, img_size, subset)


def load_cifar10(
    img_size: int = 96,
    *,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Both splits, resized and MobileNetV2-preprocessed (floats in [-1, 1])."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (
        _prep(x_train, y_train, img_size, train_subset),
        _prep(x_test, y_test, img_size, test_subset),
    )


__all__ = ["load_cifar10", "load_split", "representative_dataset"]
