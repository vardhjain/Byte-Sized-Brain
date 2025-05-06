#!/usr/bin/env python3
# train_distillbert_imdb_fast.py

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

def main():
    # 1) Load IMDb
    ds = load_dataset("imdb")

    # 2) Tokenizer (base uncased)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # 3) Preprocessing fn: truncate/pad to 128
    def preprocess(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # 4) Tokenize in parallel, remove raw text
    ds = ds.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 5) Subsample for speed (3 K train / 500 eval)
    train_ds = ds["train"].shuffle(seed=42).select(range(3000))
    eval_ds  = ds["test"].shuffle(seed=42).select(range(500))

    # 6) Load pretrained DistilBERT head
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # 7) Training arguments
    args = TrainingArguments(
        output_dir="./distilbert_imdb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,                    # remove if on CPU-only
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 8) Metric setup
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 10) Train, eval, save
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n✅ Final FP32 accuracy: {metrics['eval_accuracy']:.4f}")

    # Save both model & tokenizer for later ONNX export
    model.save_pretrained("distilbert_imdb")
    tokenizer.save_pretrained("distilbert_imdb")

if __name__ == "__main__":
    main()