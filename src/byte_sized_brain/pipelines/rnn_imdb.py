"""RNN (LSTM) on IMDB — FP32 TFLite vs dynamic-range TFLite.

The LSTM needs the TF-Select (Flex) runtime, so full-integer PTQ isn't available;
the honest quantized variant is **dynamic-range** (INT8 weights, FP32 activations).
Both variants are labelled accordingly — no "INT8" mislabelling, and no random
calibration data (see ``data/imdb.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Pipeline, Variant, export_savedmodel, tflite_benchmark

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


def _decision(output) -> int:
    return int(float(output[0]) > 0.5)


class RNNImdb(Pipeline):
    name = "rnn_imdb"
    framework = "tensorflow"

    def _params(self, cfg: PipelineConfig) -> tuple[int, int]:
        return (cfg.data.num_words or 10000, cfg.data.max_len or 200)

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        from tensorflow.keras import callbacks

        from ..data.imdb import load_imdb
        from ..models.rnn import build_lstm

        num_words, max_len = self._params(cfg)
        (x_train, y_train), (x_test, y_test) = load_imdb(num_words, max_len)
        if cfg.train.train_subset:
            x_train, y_train = x_train[: cfg.train.train_subset], y_train[: cfg.train.train_subset]

        model = build_lstm(num_words=num_words, max_len=max_len)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            x_train,
            y_train,
            validation_split=cfg.train.validation_split,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=2,
        )
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        export_savedmodel(model, paths.fp32_source)
        return float(acc)

    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        return [
            Variant("fp32", "none", "tflite", paths.tflite("fp32")),
            Variant("dynamic_range", "dynamic_range", "tflite", paths.tflite("dynamic_range")),
        ]

    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        from ..convert import tflite

        tflite.to_fp32(paths.fp32_source, paths.tflite("fp32"), flex=True)
        tflite.to_dynamic_range(paths.fp32_source, paths.tflite("dynamic_range"), flex=True)
        return self.variants(cfg, paths)

    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        from ..data.imdb import load_imdb

        num_words, max_len = self._params(cfg)
        (_, _), (x_test, y_test) = load_imdb(num_words, max_len)
        return tflite_benchmark(cfg, self.variants(cfg, paths), x_test, y_test, decision_fn=_decision)
