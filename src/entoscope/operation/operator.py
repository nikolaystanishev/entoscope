from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from entoscope.operation.input import OperationInput
from entoscope.operation.plan import OperationPlan

if TYPE_CHECKING:
    from entoscope.operation.collector import AbstractCollector


class Operator(abc.ABC):
    """Runs an operation's method over its input and hands every result to a collector.

    An operator is built from ``(model, collector_class=...)``. The collector class decides what a run
    leaves behind, which is how a run opts in or out of its HTML page. The method is the one the input's
    config names, built when the run starts.
    """

    @abc.abstractmethod
    def apply(
        self,
        operation_input: OperationInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> AbstractCollector | None:
        """Run the operation and save what it collected into ``save_dir``.

        The method is built before any forward pass, so a config that names something the run cannot do
        fails before the model is touched. Where the results were saved is logged through the
        ``entoscope`` logger.

        Args:
            operation_input: The config and prompts to run, prepared in place as the run needs.
            save_dir: The directory to write the run's files into, created if missing. ``None`` writes
                nothing.
            return_collector: Whether to return the collector.
            verbose: Whether to show a progress bar.

        Returns:
            The collector, whose ``result`` holds what the run collected, if ``return_collector`` is set,
            and ``None`` otherwise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def plan(self, operation_input: OperationInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        """Describe the run ``apply`` would make, without a forward pass and without touching ``save_dir``.

        The input's index is resolved exactly as the run resolves it, and kept: passing the same input to
        ``apply`` afterwards does not resolve it again.

        Args:
            operation_input: The config and prompts the run would use.
            save_dir: The directory the run would write into, named in the plan's outputs.
            verbose: Whether to also log the plan.

        Returns:
            The plan: the run's steps, forward and backward passes, activations, outputs and index.
        """
        raise NotImplementedError()
