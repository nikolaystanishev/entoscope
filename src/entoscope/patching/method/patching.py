import abc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Generic, cast

import torch
from typing_extensions import TypeVar

from entoscope.model import Model
from entoscope.patching.config import PatchingConfig, PatchingMethodConfig
from entoscope.patching.input import PatchingInput
from entoscope.patching.metric import AbstractComparison


@dataclass(frozen=True)
class RowResult:
    """What a method reports for one index row.

    Attributes:
        output: The result row the collector files the report under.
        metrics: The row's value in every result column, keyed by the metric's name, which is its column.
        generation: The token the patched run predicts next. Only a method that actually runs the patched
            model has one, and the gradient methods, which estimate the metric without such a run, leave it
            unset.
    """

    output: int
    metrics: dict[str, float]
    generation: int | None = None


PatchingMethodConfigT = TypeVar("PatchingMethodConfigT", bound=PatchingMethodConfig, default=PatchingMethodConfig)


class AbstractPatchingMethod(abc.ABC, Generic[PatchingMethodConfigT]):
    """The base of every patching method: how a run turns its index into one number per row.

    Every method produces the same primitive for each row, ``delta``, the metric of the patched run
    minus the metric of the unpatched target run, one per metric, and ``_report`` reports each under its
    metric's comparison. What separates the methods is only how ``delta`` is
    reached: by patching and running the model, or by estimating from gradients.

    A method is built once per run. Its type argument names the dataclass of its method config,
    extending ``PatchingMethodConfig``, which a config file's ``method`` is read into. A method naming
    none takes ``PatchingMethodConfig`` itself, which no field fits.

    Args:
        model: The model the method patches.
        config: The run's config.

    Attributes:
        batched: Whether the method runs the patched model ``batch_size`` rows at a time, so a plan
            reports its batches.
        method_config: The run's method config, typed by the method's type argument.
    """

    batched: ClassVar[bool] = False

    def __init__(self, model: Model, config: PatchingConfig):
        self.model = model
        self.config = config
        self.method_config = cast(PatchingMethodConfigT, config.method)

    @abc.abstractmethod
    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        """Score every patched run.

        Args:
            patching_input: The prepared input.
            verbose: Whether to show a progress bar.

        Yields:
            One ``RowResult`` per patched run, its metric values keyed by result column.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def passes(self, patching_input: PatchingInput, batches: int) -> tuple[int, int]:
        """Count the passes the sweep costs, over and above the input's cached runs.

        Args:
            patching_input: The input, its index resolved: ``rows()`` lists the patched runs and
                ``target_activation_names()`` the hooks a patch writes.
            batches: The number of batches the patched runs fall into, for a batched method.

        Returns:
            The forward and the backward passes.
        """
        raise NotImplementedError

    def check(self, patching_input: PatchingInput) -> None:
        """Reject an input the method cannot score, before either prompt is run.

        Called once the input's index is resolved, by ``plan`` as well as by the run, so a plan never
        promises a run the method would refuse. Nothing is rejected by default.

        Args:
            patching_input: The input, its index resolved and no forward pass made.

        Raises:
            ValueError: If the method cannot score the input.
        """

    def _report(self, patching_input: PatchingInput, deltas: dict[str, torch.Tensor]) -> dict[str, float]:
        """Report one row's change in every metric under the metric's comparison.

        Args:
            patching_input: The prepared input.
            deltas: Each metric's patched value minus its unpatched target value, keyed by the metric's name.

        Returns:
            The row's value in every result column, keyed by the metric's name, which is its column.
        """
        return {
            name: AbstractComparison.get(metric.comparison)
            .apply(patching_input.metrics[name], patching_input, deltas[name])
            .item()
            for name, metric in self.config.metric.items()
        }
