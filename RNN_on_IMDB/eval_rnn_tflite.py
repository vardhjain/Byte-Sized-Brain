#!/usr/bin/env python3
import os
import time
import psutil
import tracemalloc
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_tflite(model_path, num_samples=500):
    size_mb = os.path.getsize(model_path) / (1024**2)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    (_, _), (x_test, y_test) = imdb.load_data(num_words=10000)
    x = pad_sequences(x_test, maxlen=200).astype(np.float32)
    x_eval, y_eval = x[:num_samples], y_test[:num_samples]

    start_ram = psutil.virtual_memory().used
    tracemalloc.start()

    total_time = 0.0
    correct    = 0

    for inp_vec, lbl in zip(x_eval, y_eval):
        t0 = time.time()
        interpreter.set_tensor(inp_idx, inp_vec[None, :])
        interpreter.invoke()
        total_time += time.time() - t0

        score = interpreter.get_tensor(out_idx)[0][0]
        pred = 1 if score > 0.5 else 0
        if pred == lbl:
            correct += 1

    end_ram = psutil.virtual_memory().used
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'size_mb':        size_mb,
        'accuracy':       correct / num_samples,
        'latency_ms':     total_time / num_samples * 1000,
        'ram_delta_mb':   (end_ram - start_ram) / (1024**2),
        'peak_malloc_mb': peak / (1024**2),
    }

if __name__ == "__main__":
    models = {
        'FP32': 'imdb_rnn_fp32.tflite',
        'INT8': 'imdb_rnn_fixed.tflite',
    }

    results = {name: evaluate_tflite(path) for name, path in models.items()}
    base_size = results['FP32']['size_mb']
    for name, r in results.items():
        r['reduction_pct'] = 0.0 if name == 'FP32' else (base_size - r['size_mb']) / base_size * 100

    # Print table
    hdr = f"{'Model':<6}{'Size(MB)':>10}{'Red(%)':>10}{'Acc':>8}{'Lat(ms)':>10}{'RAM Δ':>10}{'Peak':>10}"
    print(hdr)
    for name, r in results.items():
        print(f"{name:<6}"
              f"{r['size_mb']:10.2f}"
              f"{r['reduction_pct']:10.2f}"
              f"{r['accuracy']:8.4f}"
              f"{r['latency_ms']:10.2f}"
              f"{r['ram_delta_mb']:10.2f}"
              f"{r['peak_malloc_mb']:10.2f}")