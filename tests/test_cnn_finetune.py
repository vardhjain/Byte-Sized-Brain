"""Fine-tuning must not touch BatchNorm statistics (checked on a tiny stand-in backbone)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tf


def test_unfreeze_backbone_trains_weights_but_keeps_batchnorm_frozen() -> None:
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    from byte_sized_brain.models.cnn import unfreeze_backbone

    keras.utils.set_random_seed(0)
    base = keras.Sequential(
        [
            keras.layers.Input(shape=(8, 8, 3)),
            keras.layers.Conv2D(4, 3, name="conv"),
            keras.layers.BatchNormalization(name="bn"),
            keras.layers.GlobalAveragePooling2D(),
        ]
    )
    base.trainable = False
    inputs = keras.Input(shape=(8, 8, 3))
    outputs = keras.layers.Dense(2, activation="softmax")(base(inputs, training=False))
    model = keras.Model(inputs, outputs)

    unfreeze_backbone(base)
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="sparse_categorical_crossentropy")

    bn, conv = base.get_layer("bn"), base.get_layer("conv")
    mean_before = bn.moving_mean.numpy().copy()
    kernel_before = conv.get_weights()[0].copy()

    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 2.0, (16, 8, 8, 3)).astype("float32")  # far from the initial mean of 0
    model.fit(x, rng.integers(0, 2, 16), epochs=1, batch_size=8, verbose=0)

    assert np.array_equal(mean_before, bn.moving_mean.numpy())  # statistics untouched
    assert not np.array_equal(kernel_before, conv.get_weights()[0])  # weights did fine-tune
