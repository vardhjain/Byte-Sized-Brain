"""Dataset loaders. TensorFlow/Keras is imported lazily inside each function."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def representative_dataset(x: np.ndarray, n: int = 100) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Real samples for static INT8 calibration, one batch-1 example at a time.

    Calibration has to see the distribution the model will run on. The original
    project fed ``np.random.randint`` noise here, which calibrates activation
    ranges against inputs the model never sees.
    """
    n = min(n, len(x))

    def gen() -> Iterator[list[np.ndarray]]:
        for i in range(n):
            yield [x[i : i + 1].astype(np.float32)]

    return gen


__all__ = ["representative_dataset"]
