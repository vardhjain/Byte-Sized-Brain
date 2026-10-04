"""DistilBERT on IMDB: FP32 ONNX vs dynamic INT8 ONNX.

Unlike the original project, the FP32 ONNX graph is always exported and kept, so
the comparison (and the eval) actually have both files to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Pipeline, Variant, benchmark_variants

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


class DistilBertImdb(Pipeline):
    name = "distilbert_imdb"
    framework = "pytorch"
    quantization = "dynamic_int8"

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        from ..models.distilbert import train_distilbert

        return train_distilbert(cfg, paths)

    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        return [
            Variant("fp32", "none", "onnxruntime", paths.onnx("fp32")),
            Variant("int8", self.quantization, "onnxruntime", paths.onnx("int8")),
        ]

    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        from ..convert import onnx

        seq_len = cfg.data.max_len or 128
        onnx.export_fp32(
            paths.fp32_source, paths.onnx("fp32"), seq_len=seq_len, opset=cfg.convert.opset
        )
        onnx.quantize_int8(paths.onnx("fp32"), paths.onnx("int8"))
        return self.variants(cfg, paths)

    def _eval_samples(self, cfg: PipelineConfig, paths: Paths):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(paths.fp32_source)
        seq_len = cfg.data.max_len or 128
        n = cfg.benchmark.num_samples

        ds = load_dataset(cfg.dataset, split="test").shuffle(seed=cfg.seed)
        ds = ds.select(range(min(n, len(ds))))
        labels = np.array(ds["label"], dtype=np.int64)
        enc = tokenizer(
            list(ds["text"]),
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        samples = [
            {"input_ids": ids[i : i + 1], "attention_mask": mask[i : i + 1]}
            for i in range(len(labels))
        ]
        return samples, labels

    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        samples, labels = self._eval_samples(cfg, paths)
        return benchmark_variants(
            cfg,
            paths,
            self.variants(cfg, paths),
            samples,
            labels,
            decision_fn=lambda o: int(np.argmax(o)),
        )
