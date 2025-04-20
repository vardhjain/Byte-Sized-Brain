# train_mnist_ffn.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# 1) Load & preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# 2) Define the FFN model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64,  activation="relu"),
    keras.layers.Dense(10,  activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3) Train
model.fit(x_train, y_train,
          validation_split=0.1,
          epochs=5,
          batch_size=64)

# 4) Evaluate baseline
loss, acc = model.evaluate(x_test, y_test)
print(f"Baseline FP32 accuracy: {acc:.4f}")

# 5) Save Keras model
model.export("mnist_ffn_fp32")   # exports a TF SavedModel