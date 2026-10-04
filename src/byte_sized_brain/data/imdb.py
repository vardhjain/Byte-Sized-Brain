"""IMDB loader for the LSTM pipeline (Keras integer-encoded reviews).

The LSTM is quantized with dynamic-range PTQ, which needs no calibration data.
If a static INT8 path is ever added, calibrate it with real padded sequences from
:func:`load_imdb` via ``byte_sized_brain.data.representative_dataset``, never with
random integers (the bug the original project shipped).
"""

from __future__ import annotations

import numpy as np


def pad_sequences(seqs, maxlen: int) -> np.ndarray:
    """pad_sequences moved from keras.preprocessing to keras.utils in Keras 3."""
    try:
        from tensorflow.keras.utils import pad_sequences as _pad
    except ImportError:  # older Keras
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
    return _pad(seqs, maxlen=maxlen)


def load_imdb(
    num_words: int = 10000,
    max_len: int = 200,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return padded, integer-encoded IMDB reviews as float32 (LSTM expects floats)."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    x_train = pad_sequences(x_train, max_len).astype(np.float32)
    x_test = pad_sequences(x_test, max_len).astype(np.float32)
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))
