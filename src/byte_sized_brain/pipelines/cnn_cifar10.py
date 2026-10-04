"""CNN (MobileNetV2) on CIFAR-10: FP32 vs static INT8 TFLite.

Two-stage training: train the head with the backbone frozen, then optionally
unfreeze and fine-tune at a lower learning rate (``fine_tune_epochs``). The
BatchNorm layers stay frozen during fine-tuning (see ``unfreeze_backbone`` in
``models/cnn.py``), so the ImageNet statistics are not overwritten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Pipeline, Variant, benchmark_variants, export_savedmodel

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


class CNNCifar10(Pipeline):
    name = "cnn_cifar10"
    framework = "tensorflow"
    quantization = "static_int8"

    def _img_size(self, cfg: PipelineConfig) -> int:
        return cfg.data.img_size or 96

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        from tensorflow import keras

        from ..data.cifar10 import load_cifar10
        from ..models.cnn import build_mobilenet, unfreeze_backbone

        (x_train, y_train), (x_test, y_test) = load_cifar10(
            self._img_size(cfg),
            train_subset=cfg.train.train_subset,
            test_subset=cfg.train.eval_subset,
        )

        model, base = build_mobilenet(self._img_size(cfg))
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.train.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_train,
            y_train,
            validation_split=cfg.train.validation_split,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            verbose=2,
        )

        if cfg.train.fine_tune_epochs:
            unfreeze_backbone(base)
            model.compile(
                optimizer=keras.optimizers.Adam(cfg.train.fine_tune_lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                x_train,
                y_train,
                validation_split=cfg.train.validation_split,
                epochs=cfg.train.fine_tune_epochs,
                batch_size=cfg.train.batch_size,
                verbose=2,
            )

        _, acc = model.evaluate(x_test, y_test, verbose=0)
        export_savedmodel(model, paths.fp32_source)
        return float(acc)

    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        return [
            Variant("fp32", "none", "tflite", paths.tflite("fp32")),
            Variant("int8", self.quantization, "tflite", paths.tflite("int8")),
        ]

    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        from ..convert import tflite
        from ..data import representative_dataset
        from ..data.cifar10 import load_split

        # Only preprocess the calibration images, not the whole training split.
        x_rep, _ = load_split("train", self._img_size(cfg), subset=max(cfg.convert.rep_samples, 1))
        tflite.to_fp32(paths.fp32_source, paths.tflite("fp32"))
        tflite.to_static_int8(
            paths.fp32_source,
            paths.tflite("int8"),
            representative_dataset(x_rep, cfg.convert.rep_samples),
            int8_io=cfg.convert.int8_io,
        )
        return self.variants(cfg, paths)

    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        from ..data.cifar10 import load_split

        # Only preprocess as many test images as we'll actually score.
        x_test, y_test = load_split("test", self._img_size(cfg), subset=cfg.benchmark.num_samples)
        return benchmark_variants(
            cfg,
            paths,
            self.variants(cfg, paths),
            x_test,
            y_test,
            decision_fn=lambda o: int(np.argmax(o)),
        )
