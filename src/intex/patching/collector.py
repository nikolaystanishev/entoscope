import abc
import os
from collections.abc import Sequence
from typing import Optional

import torch

from intex.patching.index import PatchingIndex
from intex.util.data import create_if_not_exists
from intex.util.executor import none_executor

CHUNK_SIZE = 256


class AbstractCollector(abc.ABC):
    @abc.abstractmethod
    def prepare(self, index: PatchingIndex):
        raise NotImplementedError

    @abc.abstractmethod
    def collect(
        self,
        source_generation: int,
        target_generation: int,
        logits: torch.Tensor,
        index: Optional[Sequence[int]] = None,
        metric_value: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, cache_path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    def __call__(
        self,
        source_generation: int,
        target_generation: int,
        logits: torch.Tensor,
        index: Optional[Sequence[int]] = None,
        metric_value: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        return self.collect(source_generation, target_generation, logits, index, metric_value, cache_path)


class PatchingCollector(AbstractCollector):
    def __init__(self):
        self.metric_output = None

        self.collector_cache = []
        self.chunk = 0

        self.nan_tensor = torch.tensor([float("nan")]).to("mps")

    def prepare(self, index: PatchingIndex):
        self.metric_output = index.prepare_output_placeholder(device="mps")

    def collect(
        self,
        source_generation: int,
        target_generation: int,
        logits: torch.Tensor,
        index: Optional[Sequence[int]] = None,
        metric_value: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        self.metric_output[index] = metric_value

        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        probabilities = logits.softmax(dim=-1)
        logits_rank = logits.argsort(dim=-1, descending=True)

        source_logits = none_executor(lambda el: logits[..., el], source_generation, self.nan_tensor)
        source_probabilities = none_executor(lambda el: probabilities[..., el], source_generation, self.nan_tensor)
        source_rank = none_executor(
            lambda el: (logits_rank == el).nonzero(as_tuple=True)[0], source_generation, self.nan_tensor
        )

        target_logits = none_executor(lambda el: logits[..., el], target_generation, self.nan_tensor)
        target_probabilities = none_executor(lambda el: probabilities[..., el], target_generation, self.nan_tensor)
        target_rank = none_executor(
            lambda el: (logits_rank == el).nonzero(as_tuple=True)[0], target_generation, self.nan_tensor
        )

        top_5 = logits.topk(5)
        top_5_logits = top_5.values.squeeze(0)
        top_5_probabilities = probabilities.topk(5)
        top_5 = top_5_probabilities.indices.squeeze(0)
        top_5_probabilities = top_5_probabilities.values.squeeze(0)

        self.collector_cache.append(
            [
                source_logits,
                target_logits,
                source_probabilities,
                target_probabilities,
                source_rank,
                target_rank,
                top_5,
                top_5_logits,
                top_5_probabilities,
                index,
            ]
        )

        if len(self.collector_cache) >= CHUNK_SIZE and cache_path is not None:
            self.save(cache_path)
            self.reset(hard=False)
            self.chunk += 1

    def save(self, cache_path: str):
        create_if_not_exists(cache_path)
        torch.save(self.collector_cache, os.path.join(cache_path, f"artefact-{str(self.chunk)}.pt"))

    def reset(self, hard: bool = True):
        del self.collector_cache
        self.collector_cache = []

        if hard:
            self.chunk = 0

            del self.metric_output
            self.metric_output = None


class EvaluationCollector(AbstractCollector):
    def __init__(self):
        self._collector_cache = []
        self._chunk_collector_cache = []

        self.nan_tensor = torch.tensor([float("nan")]).to("mps")

    def prepare(self, index: PatchingIndex):
        pass

    def collect(
        self,
        factual_generation: int,
        hallucination_generation: Optional[int],
        logits: torch.Tensor,
        index: Optional[Sequence[int]] = None,
        metric_value: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        probabilities = logits.softmax(dim=-1)

        factual_probability = none_executor(lambda el: probabilities[..., el], factual_generation, self.nan_tensor)
        hallucination_probability = none_executor(
            lambda el: probabilities[..., el], hallucination_generation, self.nan_tensor
        )

        argmax_token = probabilities.argmax(dim=-1)
        argmax_probability = probabilities[..., argmax_token]

        self._chunk_collector_cache.append(
            torch.cat(
                (factual_probability, hallucination_probability, argmax_probability.squeeze(0), argmax_token), dim=0
            )
        )

        if len(self._chunk_collector_cache) >= CHUNK_SIZE:
            self.chunk_to_collector()

    @property
    def collector_cache(self):
        if len(self._chunk_collector_cache) != 0:
            self.chunk_to_collector()
        return self._collector_cache

    def chunk_to_collector(self):
        self._collector_cache.extend(
            map(lambda el: (*el[:-1], int(el[-1])), torch.stack(self._chunk_collector_cache).tolist())
        )

    def save(self, cache_path: str):
        pass

    def reset(self, hard: bool = True):
        del self._collector_cache
        self._collector_cache = []


class LogitsCollector(AbstractCollector):
    def __init__(self):
        self._collector_cache = []

    def prepare(self, index: PatchingIndex):
        pass

    def collect(
        self,
        factual_generation: int,
        hallucination_generation: Optional[int],
        logits: torch.Tensor,
        index: Optional[Sequence[int]] = None,
        metric_value: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]

        self._collector_cache.append(logits)

    @property
    def collector_cache(self):
        return self._collector_cache

    def save(self, cache_path: str):
        pass

    def reset(self, hard: bool = True):
        del self._collector_cache
        self._collector_cache = []
