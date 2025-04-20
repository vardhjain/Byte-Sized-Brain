# eval_tflite.py

import tensorflow as tf
import numpy as np
import psutil, time

# Load interpreter
interpreter = tf.lite.Interpreter(model_path="mnist_ffn_int8.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test.astype("float32") / 255.0 * 255).astype(np.int8)  # scale to int8 range

# Run a subset for speed
num_samples = 500
x_eval = x_test[:num_samples]
y_eval = y_test[:num_samples]

# Inference loop
correct = 0
total_time = 0
start_ram = psutil.virtual_memory().used

for img, label in zip(x_eval, y_eval):
    inp = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], inp)
    t0 = time.time()
    interpreter.invoke()
    total_time += (time.time() - t0)
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    if np.argmax(pred) == label:
        correct += 1

end_ram = psutil.virtual_memory().used

print(f"INT8 TFLite accuracy: {correct/num_samples:.4f}")
print(f"Avg latency per sample: {total_time/num_samples*1000:.2f} ms")
print(f"Approx RAM delta: {(end_ram-start_ram)/1e6:.2f} MB")