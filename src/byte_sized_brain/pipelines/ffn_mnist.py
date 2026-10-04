"""FFN on MNIST: FP32 TFLite vs static INT8 TFLite (real-data calibration)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Pipeline, Variant, benchmark_variants, export_savedmodel

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


class FFNMnist(Pipeline):
    name = "ffn_mnist"
    framework = "tensorflow"
    quantization = "static_int8"

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        from tensorflow import keras

        from ..data.mnist import load_mnist
        from ..models.ffn import build_ffn

        (x_train, y_train), (x_test, y_test) = load_mnist()
        if cfg.train.train_subset:
            x_train, y_train = x_train[: cfg.train.train_subset], y_train[: cfg.train.train_subset]

        model = build_ffn()
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
        from ..data.mnist import load_mnist

        (x_train, _), _ = load_mnist()
        tflite.to_fp32(paths.fp32_source, paths.tflite("fp32"))
        tflite.to_static_int8(
            paths.fp32_source,
            paths.tflite("int8"),
            representative_dataset(x_train, cfg.convert.rep_samples),
            int8_io=cfg.convert.int8_io,
        )
        return self.variants(cfg, paths)

    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        from ..data.mnist import load_mnist

        (_, _), (x_test, y_test) = load_mnist()
        return benchmark_variants(
            cfg,
            paths,
            self.variants(cfg, paths),
            x_test,
            y_test,
            decision_fn=lambda o: int(np.argmax(o)),
        )
