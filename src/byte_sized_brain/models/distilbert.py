"""DistilBERT fine-tuning for IMDB sentiment.

Targets transformers 4.45+ where ``evaluation_strategy`` is now ``eval_strategy``
and ``Trainer(tokenizer=...)`` is now ``processing_class=...``. ``fp16`` is forced
off (this runs on CPU), TensorBoard logging is disabled (``report_to=[]``) so no
``runs/`` scratch is created, and the final model + tokenizer are saved to
``paths.fp32_source`` so the ONNX exporter has a real source to read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


def train_distilbert(cfg: PipelineConfig, paths: Paths) -> float:
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    max_len = cfg.data.max_len or 128
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    raw = load_dataset(cfg.dataset)

    def preprocess(batch: dict) -> dict:
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=max_len
        )

    ds = raw.map(
        preprocess,
        batched=True,
        num_proc=cfg.train.num_proc,
        remove_columns=["text"],
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = ds["train"].shuffle(seed=cfg.seed)
    eval_ds = ds["test"].shuffle(seed=cfg.seed)
    if cfg.train.train_subset:
        train_ds = train_ds.select(range(cfg.train.train_subset))
    if cfg.train.eval_subset:
        eval_ds = eval_ds.select(range(cfg.train.eval_subset))

    model = DistilBertForSequenceClassification.from_pretrained(cfg.model, num_labels=2)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=str(paths.dir / "hf_trainer"),
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        fp16=False,  # CPU
        dataloader_num_workers=0,  # Windows-safe
        report_to=[],  # no TensorBoard runs/ scratch
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    model.save_pretrained(paths.fp32_source)
    tokenizer.save_pretrained(paths.fp32_source)
    return float(metrics.get("eval_accuracy", float("nan")))
