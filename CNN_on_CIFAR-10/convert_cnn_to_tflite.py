import os, tensorflow as tf, numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

# 1) Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_cifar10_fp32")

# 2) INT8 quant
converter.optimizations = [tf.lite.Optimize.DEFAULT]
(x_train, _), _ = cifar10.load_data()
IMG_SIZE=96
x_train = tf.image.resize(x_train,(IMG_SIZE,IMG_SIZE)).numpy()
x_train = preprocess_input(x_train)
def rep_data_gen():
    for img in x_train[:100]:
        yield [np.expand_dims(img,axis=0)]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

# 3) Convert & save
tflite_model = converter.convert()
open("mobilenetv2_cifar10_int8.tflite","wb").write(tflite_model)

# 4) Size stats
def folder_size(p):
    return sum(os.path.getsize(os.path.join(dp,f))
               for dp,_,files in os.walk(p) for f in files)/1e6
fp32 = folder_size("mobilenetv2_cifar10_fp32")
int8 = os.path.getsize("mobilenetv2_cifar10_int8.tflite")/1e6
print(f"FP32 SavedModel: {fp32:.2f}MB\nINT8 TFLite: {int8:.2f}MB\nReduction: {(fp32-int8)/fp32*100:.2f}%")