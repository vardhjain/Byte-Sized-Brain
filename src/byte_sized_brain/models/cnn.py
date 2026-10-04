"""MobileNetV2 backbone + classification head for CIFAR-10."""

from __future__ import annotations

from typing import Any


def build_mobilenet(img_size: int = 96, num_classes: int = 10) -> tuple[Any, Any]:
    """Return ``(model, base)``; ``base`` is exposed so the pipeline can unfreeze it.

    While ``base.trainable`` is False every backbone layer, BatchNorm included,
    runs in inference mode. Use :func:`unfreeze_backbone` to fine-tune it.
    """
    from tensorflow import keras

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_cifar10"), base


def unfreeze_backbone(base: Any) -> None:
    """Make the backbone weights trainable while keeping BatchNorm frozen.

    Fine-tuning on small CIFAR batches must not overwrite the ImageNet BatchNorm
    statistics, which is the classic way transfer learning wrecks its own accuracy.
    In Keras 3 a ``training=False`` argument on the nested backbone call is not
    enough, because ``fit`` passes ``training=True`` down to every layer of a
    functional model. A layer with ``trainable = False`` always runs in inference
    mode, so the BatchNorm layers are frozen one by one instead.
    """
    from tensorflow import keras

    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
