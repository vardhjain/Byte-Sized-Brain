# convert_rnn_to_tflite_fp32.py

import tensorflow as tf

# 1) Load your SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("imdb_rnn_fp32")

# 2) Enable Flex (SELECT_TF_OPS) so TF-only ops like TensorListReserve / LSTM are supported
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # normal built-in TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS          # TensorFlow-select (Flex) ops
]
# disable the lowering of TensorList operators (we want to keep them as TF-Select)
converter._experimental_lower_tensor_list_ops = False

# 3) No quantization here: pure FP32 TFLite
tflite_model = converter.convert()

# 4) Write it out
with open("imdb_rnn_fp32.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Wrote imdb_rnn_fp32.tflite with Flex ops enabled")