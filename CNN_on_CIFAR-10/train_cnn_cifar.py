import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1) Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.squeeze(), y_test.squeeze()

# 2) Resize → 96×96
IMG_SIZE = 96
x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy()
x_test  = tf.image.resize(x_test,  (IMG_SIZE, IMG_SIZE)).numpy()

# 3) Preprocess
x_train = preprocess_input(x_train)
x_test  = preprocess_input(x_test)

# 4) MobilenetV2 backbone
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE,IMG_SIZE,3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base.trainable = False

# 5) Head
model = keras.Sequential([
    base,
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])

# 6) Train head
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128)

# 7) Fine-tune
base.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=128)

# 8) Eval & save
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final FP32 accuracy: {acc:.4f}")
model.export("mobilenetv2_cifar10_fp32", save_format="tf")