#!/usr/bin/env python3
import os
import time
import tracemalloc
import psutil
import onnxruntime as ort
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def load_tokenizer(path_or_dir):
    if os.path.isdir(path_or_dir):
        return PreTrainedTokenizerFast.from_pretrained(path_or_dir, local_files_only=True)
    elif os.path.isfile(path_or_dir):
        return PreTrainedTokenizerFast(
            tokenizer_file=path_or_dir,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]"
        )
    else:
        raise FileNotFoundError(f"No tokenizer found at {path_or_dir}")

def prepare_data(tokenizer, max_length=128, num_samples=500):
    # Load the public IMDB test split
    ds = load_dataset("imdb", split="test")
    texts  = ds["text"][:num_samples]
    labels = np.array(ds["label"][:num_samples])
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"
    )
    return encodings, labels

def evaluate_onnx(model_path, encodings, labels):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names  = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    N = len(labels)

    # Model size
    size_mb = os.path.getsize(model_path) / (1024**2)

    # Warm-up
    _ = sess.run(
        output_names,
        {n: encodings[n][0:1] for n in input_names}
    )

    # Metrics
    correct    = 0
    total_time = 0.0
    proc       = psutil.Process(os.getpid())
    start_mem  = proc.memory_info().rss
    tracemalloc.start()

    for i in range(N):
        inp = {n: encodings[n][i : i+1] for n in input_names}
        t0 = time.time()
        outs = sess.run(output_names, inp)
        total_time += time.time() - t0

        # assume first output is logits
        pred = np.argmax(outs[0], axis=-1)[0]
        if pred == labels[i]:
            correct += 1

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_mem = proc.memory_info().rss

    return {
        "size_mb":        size_mb,
        "accuracy":       correct / N,
        "latency_ms":     total_time / N * 1000,
        "ram_delta_mb":   (end_mem - start_mem) / 1e6,
        "peak_malloc_mb": peak / (1024**2),
    }

def main():
    # find your FP32 & INT8 .onnx in cwd
    models = {}
    for fn in os.listdir("."):
        if fn.endswith(".onnx"):
            l = fn.lower()
            if "fp32" in l: models["FP32"] = fn
            if "int8" in l: models["INT8"] = fn

    if set(models.keys()) != {"FP32", "INT8"}:
        print("❌ Need both FP32 and INT8 .onnx in this folder!")
        return

    # tokenizer dir must contain tokenizer.json etc.
    tokenizer = load_tokenizer("distilbert_imdb")

    encodings, labels = prepare_data(tokenizer)

    results = {k: evaluate_onnx(v, encodings, labels)
               for k, v in models.items()}

    # print table
    print("\nSummary (FP32 vs. INT8):")
    hdr = f"{'Metric':<15}{'FP32':>10}{'INT8':>10}{'Δ INT8–FP32':>15}"
    print(hdr)
    print("-"*len(hdr))
    for metric in ["size_mb","accuracy","latency_ms","ram_delta_mb","peak_malloc_mb"]:
        a = results["FP32"][metric]
        b = results["INT8"][metric]
        d = b - a
        print(f"{metric:<15}{a:>10.4f}{b:>10.4f}{d:>+15.4f}")

if __name__ == "__main__":
    main()