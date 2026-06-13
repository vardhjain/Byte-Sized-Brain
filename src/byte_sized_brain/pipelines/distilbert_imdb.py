"""DistilBERT on IMDB — FP32 ONNX vs dynamic INT8 ONNX.

Unlike the original project, the FP32 ONNX graph is always exported and kept, so
the comparison (and the eval) actually have both files to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..benchmark import OnnxRunner, benchmark_inference, build_row
from ..utils import get_logger, size_mb
from .base import Pipeline, Variant

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


class DistilBertImdb(Pipeline):
    name = "distilbert_imdb"
    framework = "pytorch"

    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        from ..models.distilbert import train_distilbert

        return train_distilbert(cfg, paths)

    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        return [
            Variant("fp32", "none", "onnxruntime", paths.onnx("fp32")),
            Variant("int8", "dynamic_int8", "onnxruntime", paths.onnx("int8")),
        ]

    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        from ..convert import onnx

        seq_len = cfg.data.max_len or 128
        onnx.export_fp32(paths.fp32_source, paths.onnx("fp32"), seq_len=seq_len, opset=cfg.convert.opset)
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
        log = get_logger(cfg.name)
        samples, labels = self._eval_samples(cfg, paths)
        rows: list[dict[str, Any]] = []
        for v in self.variants(cfg, paths):
            if not v.path.exists():
                raise FileNotFoundError(
                    f"Missing artifact {v.path}. Run `bsb convert {cfg.name}` first."
                )
            runner = OnnxRunner(v.path)
            metrics = benchmark_inference(
                runner.predict,
                samples,
                labels,
                num_samples=cfg.benchmark.num_samples,
                warmup=cfg.benchmark.warmup,
                decision_fn=lambda o: int(np.argmax(o)),
            )
            mb = size_mb(v.path)
            rows.append(
                build_row(
                    cfg,
                    variant=v.name,
                    quantization=v.quantization,
                    runtime=v.runtime,
                    size_mb=mb,
                    metrics=metrics,
                )
            )
            log.info(
                "%-6s acc=%.4f lat=%.2fms (p95 %.2f) size=%.2fMB",
                v.name,
                metrics["accuracy"],
                metrics["latency_ms_mean"],
                metrics["latency_ms_p95"],
                mb,
            )
        return rows
