"""IMDB loader for the LSTM pipeline (Keras integer-encoded reviews).

Note the representative dataset here uses **real, padded IMDB sequences**. The
original project fed ``np.random.randint`` noise as calibration data, which makes
post-training quantization calibrate against a distribution the model never sees
— a no-op at best. This is the corrected, honest version.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def _pad_sequences(seqs, maxlen: int):
    """pad_sequences moved from keras.preprocessing to keras.utils in Keras 3."""
    try:
        from tensorflow.keras.utils import pad_sequences
    except ImportError:  # older Keras
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(seqs, maxlen=maxlen)


def load_imdb(
    num_words: int = 10000,
    max_len: int = 200,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return padded, integer-encoded IMDB reviews as float32 (LSTM expects floats)."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    x_train = _pad_sequences(x_train, max_len).astype(np.float32)
    x_test = _pad_sequences(x_test, max_len).astype(np.float32)
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))


def representative_dataset(x: np.ndarray, n: int = 100) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Real padded IMDB sequences for calibration (NOT random noise)."""
    n = min(n, len(x))

    def gen() -> Iterator[list[np.ndarray]]:
        for i in range(n):
            yield [x[i : i + 1].astype(np.float32)]

    return gen
