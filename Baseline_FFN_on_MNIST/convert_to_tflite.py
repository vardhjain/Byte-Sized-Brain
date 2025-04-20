# convert_to_tflite.py

import tensorflow as tf
import numpy as np
import os

# Load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("mnist_ffn_fp32")

# Set optimization flag for full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide a representative dataset generator
def representative_data_gen():
    for _ in range(100):
        # random sample from train set
        data = np.random.rand(1,28,28).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_data_gen

# Specify supported ops—ensure full integer inference
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
# Set input/output to int8
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

# Convert & save
tflite_model = converter.convert()
open("mnist_ffn_int8.tflite", "wb").write(tflite_model)

# Print file sizes
print("FP32 SavedModel size:",
      sum(os.path.getsize(os.path.join(dirpath, f))
          for dirpath,_,files in os.walk("mnist_ffn_fp32")
          for f in files)/1e6, "MB")
print("INT8 TFLite size:",
      os.path.getsize("mnist_ffn_int8.tflite")/1e6, "MB")