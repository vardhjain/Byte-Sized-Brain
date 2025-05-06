import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Load & preprocess MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)

# 2) Build a simple feed-forward network
model = keras.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 3) Train
model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
)

# 4) Final FP32 eval
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Baseline FP32 accuracy: {acc:.4f}")

# 5) Export as SavedModel for TFLite
model.export("mnist_ffn_fp32")