#!/usr/bin/env python3
# convert_distillbert_to_onnx.py

import os
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_onnx(model_dir, output_fp32, output_int8):
    # Load fine-tuned model
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Load tokenizer: use your saved version if present, else fall back to base
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    except OSError:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Create a dummy input
    dummy = tokenizer(
        "This is a sample input for ONNX export.",
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    # 1) Export FP32 ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    # 2) Quantize weights to INT8
    quantize_dynamic(output_fp32, output_int8, weight_type=QuantType.QInt8)

    # 3) Print file‐size comparison
    fp32_size = os.path.getsize(output_fp32) / (1024**2)
    int8_size = os.path.getsize(output_int8)  / (1024**2)
    print(f"\nFP32 ONNX size : {fp32_size:.2f} MB")
    print(f"INT8 ONNX size : {int8_size:.2f} MB")
    print(f"Reduction      : {(fp32_size - int8_size) / fp32_size * 100:.2f}%")

if __name__ == "__main__":
    export_onnx(
        model_dir="./distilbert_imdb",
        output_fp32="distilbert_imdb_fp32.onnx",
        output_int8 ="distilbert_imdb_int8.onnx"
    )
    
# at the end of train_distillbert_imdb.py or convert_distilbert_to_onnx.py

from transformers import DistilBertTokenizerFast

# (re-)load the tokenizer you used for training
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# or, if you fine-tuned in memory:
#    tokenizer = your_trainer.tokenizer

# now save it to a folder called distilbert_imdb
tokenizer.save_pretrained("distilbert_imdb")
