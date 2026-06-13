"""Maps a pipeline name to its implementation. Cheap to import (no TF/torch)."""

from __future__ import annotations

from .pipelines.base import Pipeline
from .pipelines.cnn_cifar10 import CNNCifar10
from .pipelines.distilbert_imdb import DistilBertImdb
from .pipelines.ffn_mnist import FFNMnist
from .pipelines.rnn_imdb import RNNImdb

_REGISTRY: dict[str, Pipeline] = {
    p.name: p for p in (FFNMnist(), CNNCifar10(), RNNImdb(), DistilBertImdb())
}


def all_names() -> list[str]:
    return list(_REGISTRY)


def get_pipeline(name: str) -> Pipeline:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown pipeline '{name}'. Choices: {', '.join(_REGISTRY)}") from None
