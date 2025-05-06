# eval_fnn_metrics.py

import os
import warnings
import time

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Suppress oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", ".*Interpreter is deprecated.*", module="tensorflow")

def evaluate_tflite(model_path, num_samples=1000):
    # --- 1) Load the TFLite model ---
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # --- 2) Pull out dtype + quant params ---
    dtype = inp['dtype']
    scale, zero_point = inp.get('quantization', (1.0, 0))

    # --- 3) Prepare MNIST test set ---
    (_, _), (x_test, y_test) = mnist.load_data()
    x = x_test.astype(np.float32) / 255.0        # normalize to [0,1]
    y = y_test

    # flatten each image to 784 vector
    x = x.reshape(-1, 28 * 28)

    x_eval = x[:num_samples]
    y_eval = y[:num_samples]

    # --- 4) Run inference ---
    correct = 0
    total_time = 0.0
    ram_start = psutil.virtual_memory().used

    for img, label in zip(x_eval, y_eval):
        # quantize if needed
        if np.issubdtype(dtype, np.integer):
            q = img / scale + zero_point
            inp_data = np.expand_dims(q.astype(dtype), axis=0)
        else:
            inp_data = np.expand_dims(img.astype(dtype), axis=0)

        interpreter.set_tensor(inp['index'], inp_data)
        t0 = time.time()
        interpreter.invoke()
        total_time += (time.time() - t0)

        output = interpreter.get_tensor(out['index'])[0]
        pred = np.argmax(output)
        if pred == label:
            correct += 1

    ram_end = psutil.virtual_memory().used

    # --- 5) Gather metrics ---
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    return {
        'Size (MB)': size_mb,
        'Accuracy': correct / num_samples,
        'Latency (ms)': total_time / num_samples * 1000,
        'RAM Δ (MB)': (ram_end - ram_start) / 1e6
    }

if __name__ == "__main__":
    models = {
        'FP32': 'mnist_ffn_fp32.tflite',
        'INT8': 'mnist_ffn_int8.tflite'
    }

    results = {name: evaluate_tflite(path) for name, path in models.items()}

    # compute size reduction
    fp32_sz = results['FP32']['Size (MB)']
    int8_sz = results['INT8']['Size (MB)']
    results['INT8']['Size Reduction (%)'] = (fp32_sz - int8_sz) / fp32_sz * 100

    # --- 6) Print comparison table ---
    headers = ["Model", "Size", "Accuracy", "Latency", "RAM Δ", "Size ↓%"]
    fmt = "{:<6} {:>8} {:>9} {:>9} {:>8} {:>10}"
    print("\n" + fmt.format(*headers))
    print("-" * 60)
    for name, m in results.items():
        print(fmt.format(
            name,
            f"{m['Size (MB)']:.2f}MB",
            f"{m['Accuracy']:.4f}",
            f"{m['Latency (ms)']:.2f}",
            f"{m['RAM Δ (MB)']:.2f}",
            f"{m.get('Size Reduction (%)', 0):.1f}%"
        ))