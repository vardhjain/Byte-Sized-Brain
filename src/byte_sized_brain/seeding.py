"""Global determinism.

Call :func:`seed_everything` at the top of every train/convert/benchmark run.
Framework imports are lazy and best-effort so seeding works even when only one
of TensorFlow / PyTorch is installed.
"""

from __future__ import annotations

import contextlib
import os
import random


def seed_everything(seed: int = 42, *, deterministic_ops: bool = False) -> int:
    """Seed Python, NumPy, TensorFlow and PyTorch (whichever are importable)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        # set_random_seed seeds python+numpy+tf in one call (Keras 3 / TF 2.16+).
        tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            # not all ops support determinism; don't fail the run
            with contextlib.suppress(Exception):
                tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_ops:
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
