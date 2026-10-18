import logging

from entoscope.model import Model
from entoscope.operation.operator import Operator
from entoscope.operation.plan import OperationPlan
from entoscope.patching.collector import PatchingCollector, PatchingVisualizationCollector
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.factory import PatchingMethodFactory
from entoscope.util.collections import batch_generator

logger = logging.getLogger(__name__)


class Patcher(Operator):
    """Patches the source run into the target run and reports how the metrics moved, one row per patched run.

    Backends round differently at different batch shapes, so the cached source and target runs are made
    at the same width as the patched runs.

    Args:
        model: The model to patch.
        collector_class: The collector to write the results with. The default also draws
            ``result.html``, and ``PatchingCollector`` writes the CSV files alone.
        batch_size: How many patched runs one forward pass of ``exact`` holds. The gradient methods score
            the whole index from a pass per interpolation step and ignore it.
    """

    def __init__(
        self,
        model: Model,
        collector_class: type[PatchingCollector] = PatchingVisualizationCollector,
        batch_size: int = 1,
    ):
        self.model = model
        self.collector_class = collector_class
        self.batch_size = batch_size

    def apply(
        self,
        patching_input: PatchingInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> PatchingCollector | None:
        """Patch every row of the input's index and read the metrics off each patched run.

        The method is built before any forward pass, so a config that names something the run cannot do
        fails before the model is touched. Where the results were saved is logged through the
        ``entoscope`` logger.

        Args:
            patching_input: The config, prompts and answer tokens to run, prepared in place.
            save_dir: The directory to write ``result.csv``, one ``index-{i}.csv`` per patch element,
                ``config.json`` and the page into, created if missing. ``None`` writes nothing.
            return_collector: Whether to return the collector.
            verbose: Whether to show a progress bar.

        Returns:
            The collector, whose ``result`` holds every patched run's metric columns, if
            ``return_collector`` is set, and ``None`` otherwise.
        """
        collector = self._patch(patching_input, save_dir, verbose)
        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, patching_input: PatchingInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        """Describe the run ``apply`` would make, without a forward pass and without touching ``save_dir``.

        The input's index is resolved exactly as the run resolves it, and kept for ``apply``.

        Args:
            patching_input: The config, prompts and answer tokens the run would use.
            save_dir: The directory the run would write into.
            verbose: Whether to also log the plan.

        Returns:
            The plan: one step per patched run, and the forward and backward passes the method makes on
            top of one cached run per prompt.
        """
        method = PatchingMethodFactory.create(self.model, patching_input.config)
        patching_input.prepare(self.model, index_only=True)

        steps = len(patching_input.rows())
        batches = len(list(batch_generator(list(range(steps)), self.batch_size)))
        forward_passes, backward_passes = method.passes(steps, batches)

        prompts = [("target", patching_input.target_prompt, patching_input.target_tokens.shape[1])]
        if patching_input.source_prompt is not None:
            prompts.insert(0, ("source", patching_input.source_prompt, patching_input.source_tokens.shape[1]))

        plan = OperationPlan(
            operation="Patcher",
            method=patching_input.config.method.type,
            config=patching_input.config,
            steps=steps,
            steps_label="patch run",
            forward_passes=forward_passes + len(prompts),
            backward_passes=backward_passes,
            batches=batches if method.batched else None,
            index=[idx.to_df() for idx in patching_input.index],
            prompts=prompts,
            activations=patching_input.activation_names(),
            save_dir=save_dir,
            outputs=self.collector_class(patching_input, self.model, save_dir).output_files(),
        )

        if verbose:
            logger.info("%s", plan)
        return plan

    def _patch(self, patching_input: PatchingInput, save_dir: str | None, verbose: bool = True) -> PatchingCollector:
        method = PatchingMethodFactory.create(self.model, patching_input.config)
        patching_input.prepare(self.model, self.batch_size if method.batched else 1)
        collector = self.collector_class(patching_input, self.model, save_dir)

        for row in method.scores(patching_input, verbose):
            collector.collect(row)

        return collector
