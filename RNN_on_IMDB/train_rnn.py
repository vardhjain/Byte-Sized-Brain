import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# 1) Load IMDB
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
maxlen=200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=maxlen)

# 2) Build LSTM
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=maxlen),
    layers.LSTM(32),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid', 
                 kernel_regularizer=regularizers.l2(0.001))
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)

# 3) Train
model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=5,
          batch_size=32,
          callbacks=[early_stop])

# 4) Eval & save
loss, acc = model.evaluate(x_test, y_test)
print(f"Baseline FP32 accuracy: {acc:.4f}")
model.export("imdb_rnn_fp32")