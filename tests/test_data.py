"""Framework-free checks on the shared calibration helper."""

from __future__ import annotations

import numpy as np

from byte_sized_brain.data import representative_dataset


def test_representative_dataset_yields_real_batch_one_float32_samples() -> None:
    x = np.arange(40, dtype=np.float64).reshape(10, 4)
    batches = list(representative_dataset(x, n=3)())

    assert len(batches) == 3
    for i, batch in enumerate(batches):
        assert isinstance(batch, list) and len(batch) == 1
        assert batch[0].shape == (1, 4)
        assert batch[0].dtype == np.float32
        # The values are the real rows, in order (never random noise).
        assert np.array_equal(batch[0][0], x[i].astype(np.float32))


def test_representative_dataset_is_capped_to_the_data_and_is_reusable() -> None:
    x = np.zeros((2, 3), dtype=np.float32)
    gen = representative_dataset(x, n=100)
    assert len(list(gen())) == 2
    assert len(list(gen())) == 2  # the converter may iterate it more than once
