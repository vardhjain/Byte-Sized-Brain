"""LSTM sentiment classifier for IMDB.

``input_length`` was removed from ``Embedding`` in Keras 3; the sequence length
is fixed by the explicit ``Input(shape=(max_len,))`` instead.
"""

from __future__ import annotations


def build_lstm(
    num_words: int = 10000,
    max_len: int = 200,
    embed_dim: int = 64,
    lstm_units: int = 32,
):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    return keras.Sequential(
        [
            layers.Input(shape=(max_len,)),
            layers.Embedding(input_dim=num_words, output_dim=embed_dim),
            layers.LSTM(lstm_units),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(
                1,
                activation="sigmoid",
                kernel_regularizer=regularizers.l2(0.001),
            ),
        ],
        name="rnn_imdb",
    )
