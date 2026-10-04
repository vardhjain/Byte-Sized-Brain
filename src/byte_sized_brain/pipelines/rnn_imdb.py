"""RNN (LSTM) on IMDB: FP32 TFLite vs dynamic-range TFLite.

The SavedModel is exported with a **static batch dimension** so the converter can
lower the LSTM to builtin ops (a ``WHILE`` loop whose body is the LSTM cell written
out as FULLY_CONNECTED, LOGISTIC, TANH, MUL and ADD). It is not the fused
``UnidirectionalSequenceLSTM`` op, but it runs on the plain TFLite/LiteRT
interpreter (and on ARM) with no TF-Select/Flex delegate.

Full-integer PTQ isn't well-supported for LSTM, so the honest quantized variant
is **dynamic-range** (INT8 weights, FP32 activations), labelled as such rather
than mislabelled "INT8". Dynamic-range quantization needs no calibration data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Pipeline, Variant, benchmark_variants, export_savedmodel

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


def _decision(output) -> int:
    return int(float(output[0]) > 0.5)


class RNNImdb(Pipeline):
    name = "rnn_imdb"
    framework = "tensorflow"
    quantization = "dynamic_range"

    def _params(self, cfg: PipelineConfig) -> tuple[int, int]:
        return (cfg.data.num_words or 10000, cfg.data.max_len or 200)

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        import tensorflow as tf
        from tensorflow import keras

        from ..data.imdb import load_imdb
        from ..models.rnn import build_lstm

        num_words, max_len = self._params(cfg)
        (x_train, y_train), (x_test, y_test) = load_imdb(num_words, max_len)
        if cfg.train.train_subset:
            x_train, y_train = x_train[: cfg.train.train_subset], y_train[: cfg.train.train_subset]

        model = build_lstm(num_words=num_words, max_len=max_len)
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.train.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_train,
            y_train,
            validation_split=cfg.train.validation_split,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
            verbose=2,
        )
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        # Static batch=1 signature -> a loop of builtin ops (no Flex delegate needed).
        export_savedmodel(
            model,
            paths.fp32_source,
            input_signature=[tf.TensorSpec(shape=(1, max_len), dtype=tf.float32)],
        )
        return float(acc)

    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        return [
            Variant("fp32", "none", "tflite", paths.tflite("fp32")),
            Variant("dynamic_range", self.quantization, "tflite", paths.tflite("dynamic_range")),
        ]

    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        from ..convert import tflite

        # The static-batch export lets the LSTM convert to builtins, so no Flex is needed.
        tflite.to_fp32(paths.fp32_source, paths.tflite("fp32"))
        tflite.to_dynamic_range(paths.fp32_source, paths.tflite("dynamic_range"))
        return self.variants(cfg, paths)

    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        from ..data.imdb import load_imdb

        num_words, max_len = self._params(cfg)
        (_, _), (x_test, y_test) = load_imdb(num_words, max_len)
        return benchmark_variants(
            cfg, paths, self.variants(cfg, paths), x_test, y_test, decision_fn=_decision
        )
