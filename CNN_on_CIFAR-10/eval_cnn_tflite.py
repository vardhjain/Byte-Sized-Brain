# eval_cnn_tflite.py

import os, time, psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── Silence oneDNN and TFLite deprecation warnings ───────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", ".*Interpreter is deprecated.*", module="tensorflow")

def load_cifar(img_size):
    (_, _), (x, y) = cifar10.load_data()
    x = x.astype(np.float32)
    x = tf.image.resize(x, (img_size, img_size)).numpy()
    x = preprocess_input(x)  # floats in [-1,1]
    return x, y.squeeze()

def evaluate_tflite(model_path, x_data, y_data, N=1000):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    dtype = inp_det['dtype']
    qp = inp_det.get('quantization_parameters', None)
    if qp and len(qp['scales']) > 0:
        scale, zp = qp['scales'][0], qp['zero_points'][0]
    else:
        scale = zp = None

    def prep(img):
        if np.issubdtype(dtype, np.integer):
            # quantized model
            q = img / scale + zp
            return np.round(q).astype(dtype)
        else:
            # float model
            return img.astype(dtype)

    ram_before = psutil.virtual_memory().used
    t_total = 0.0
    correct = 0

    for i in range(min(N, len(x_data))):
        inp_tensor = np.expand_dims(prep(x_data[i]), 0)
        interp.set_tensor(inp_det['index'], inp_tensor)
        t0 = time.time()
        interp.invoke()
        t_total += time.time() - t0

        out = interp.get_tensor(out_det['index'])[0]
        if np.argmax(out) == int(y_data[i]):
            correct += 1

    ram_after = psutil.virtual_memory().used

    return {
        'size_mb': os.path.getsize(model_path) / (1024**2),
        'acc': correct / N,
        'lat_ms': t_total / N * 1000,
        'ram_delta_mb': (ram_after - ram_before) / 1e6
    }

if __name__ == "__main__":
    IMG_SIZE = 96
    x_test, y_test = load_cifar(IMG_SIZE)

    models = {
        'FP32': 'mobilenetv2_cifar10_fp32.tflite',
        'INT8': 'mobilenetv2_cifar10_int8.tflite',
    }

    results = {}
    for name, path in models.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"\n>>> Evaluating {name} model")
        results[name] = evaluate_tflite(path, x_test, y_test)

    # Compute INT8 size reduction vs FP32
    sz_fp32 = results['FP32']['size_mb']
    sz_int8 = results['INT8']['size_mb']
    results['INT8']['size_red%'] = 100 * (sz_fp32 - sz_int8) / sz_fp32

    # Print comparison table
    print("\n{:<6} {:>8} {:>6} {:>8} {:>10} {:>8}".format(
        "Model", "SizeMB", "Acc", "Lat_ms", "RAM Δ(MB)", "Size↓%"
    ))
    print("-" * 60)
    for name in ['FP32', 'INT8']:
        r = results[name]
        print("{:<6} {:8.2f} {:6.4f} {:8.2f} {:10.2f} {:7.1f}%".format(
            name,
            r['size_mb'],
            r['acc'],
            r['lat_ms'],
            r['ram_delta_mb'],
            r.get('size_red%', 0.0)
        ))