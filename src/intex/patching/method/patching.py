import abc
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from intex.model import Model
from intex.patching.config import PatchingConfig, PatchingMetric, metric_column
from intex.patching.input import PatchingInput
from intex.patching.metric import AbstractComparison


@dataclass(frozen=True)
class RowResult:
    """What a method reports for one index row, filed by the collector under ``output``.

    ``generation`` is the token the patched run predicts next. Only a method that actually runs the
    patched model has one; the gradient methods estimate the metric without such a run and leave it unset.
    """

    output: int
    metrics: dict[str, float]
    generation: int | None = None


class AbstractPatchingMethod(abc.ABC):
    """How a run turns its index into one number per row.

    Every method produces the same primitive for each row — ``delta``, the metric of the patched
    run minus the metric of the unpatched target run, one per metric — and ``_report`` fans it out
    into the columns the config's comparisons ask for. What separates the methods is only how
    ``delta`` is reached: by patching and running the model, or by estimating from gradients.
    """

    def __init__(self, model: Model, config: PatchingConfig, batch_size: int = 1):
        self.model = model
        self.config = config
        self.batch_size = batch_size

    @abc.abstractmethod
    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        """One ``RowResult`` per index row, its metric values keyed by result column."""
        raise NotImplementedError

    @abc.abstractmethod
    def passes(self, runs: int, batches: int) -> tuple[int, int]:
        """The (forward, backward) passes the sweep costs, over and above the input's cached runs."""
        raise NotImplementedError

    def _report(self, patching_input: PatchingInput, deltas: dict[PatchingMetric, torch.Tensor]) -> dict[str, float]:
        return {
            metric_column(metric, comparison): AbstractComparison.get(comparison)
            .apply(metric, patching_input, deltas[metric])
            .item()
            for metric, comparisons in self.config.metric.items()
            for comparison in comparisons
        }
