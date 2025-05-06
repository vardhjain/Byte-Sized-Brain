import tensorflow as tf, numpy as np, os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

# 1) FP32 → TFLite (dynamic-range quant)
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model("imdb_rnn_fp32")
converter_fp32.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp32.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter_fp32._experimental_lower_tensor_list_ops = False
def rep_gen():
    for _ in range(100):
        dummy = np.random.randint(0,10000,(1,200)).astype(np.float32)
        yield [dummy]
converter_fp32.representative_dataset = rep_gen

tflite_model = converter_fp32.convert()
open("imdb_rnn_fixed.tflite","wb").write(tflite_model)

# 2) Sizes
def folder_size(p):
    return sum(os.path.getsize(os.path.join(dp,f))
               for dp,_,files in os.walk(p) for f in files)/1e6
fp32 = folder_size("imdb_rnn_fp32"); int8=os.path.getsize("imdb_rnn_fixed.tflite")/1e6
print(f"FP32 SavedModel: {fp32:.2f}MB\nQuant TFLite: {int8:.2f}MB\nReduction: {(fp32-int8)/fp32*100:.2f}%")