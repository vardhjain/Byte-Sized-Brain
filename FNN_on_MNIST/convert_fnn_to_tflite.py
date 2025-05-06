import os
import tensorflow as tf
import numpy as np

# 0) silence oneDNN notices
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1) Convert FP32 SavedModel → FP32 TFLite
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model("mnist_ffn_fp32")
tflite_fp32 = converter_fp32.convert()
with open("mnist_ffn_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

# 2) Convert → INT8-quantized TFLite (weights+activations)
converter_int8 = tf.lite.TFLiteConverter.from_saved_model("mnist_ffn_fp32")
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

# representative dataset
from tensorflow.keras.datasets import mnist
(x_rep, _), _ = mnist.load_data()
x_rep = x_rep.astype("float32")/255.0
x_rep = x_rep.reshape(-1, 28*28)

def rep_data_gen():
    for i in range(100):
        yield [x_rep[i].reshape(1, -1)]

converter_int8.representative_dataset = rep_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type  = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()
with open("mnist_ffn_int8.tflite", "wb") as f:
    f.write(tflite_int8)

# 3) Print size comparison
def size_mb(path): return os.path.getsize(path)/(1024**2)
fp32_sz = size_mb("mnist_ffn_fp32.tflite")
int8_sz = size_mb("mnist_ffn_int8.tflite")
print(f"FP32 TFLite size: {fp32_sz:.2f} MB")
print(f"INT8 TFLite size: {int8_sz:.2f} MB")
print(f"Size reduction: {(fp32_sz-int8_sz)/fp32_sz*100:.2f}%")