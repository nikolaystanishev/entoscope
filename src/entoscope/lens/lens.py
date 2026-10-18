import logging

from tqdm import tqdm

from entoscope.lens.collector import LensCollector, LensVisualizationCollector
from entoscope.lens.input import LensInput
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.model import Model
from entoscope.operation.operator import Operator
from entoscope.operation.plan import OperationPlan

logger = logging.getLogger(__name__)


class Lens(Operator):
    """Reads every index element out onto the vocabulary through the lens ``config.method`` names.

    One forward pass caches the activations the index reads, none at all for a lens that reads only
    the weights, and every row becomes its element's top-k tokens.

    Args:
        model: The model to read.
        collector_class: The collector to write the results with. The default also draws
            ``result.html``, and ``LensCollector`` writes the CSV files alone.
    """

    def __init__(self, model: Model, collector_class: type[LensCollector] = LensVisualizationCollector):
        self.model = model
        self.collector_class = collector_class

    def apply(
        self,
        lens_input: LensInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> LensCollector | None:
        """Read every row of the input's index out onto the vocabulary.

        The method is built before any forward pass, so a config that names something the run cannot do
        fails before the model is touched. Where the results were saved is logged through the
        ``entoscope`` logger.

        Args:
            lens_input: The config and prompt to run, prepared in place.
            save_dir: The directory to write ``result.csv``, ``index.csv``, ``config.json`` and the page
                into, created if missing. ``None`` writes nothing.
            return_collector: Whether to return the collector.
            verbose: Whether to show a progress bar.

        Returns:
            The collector, whose ``result`` holds every row's top-k tokens and probabilities, if
            ``return_collector`` is set, and ``None`` otherwise.
        """
        method = LensMethodFactory.create(self.model, lens_input.config)
        lens_input.prepare(self.model)
        collector = self.collector_class(lens_input, self.model, save_dir)

        for single_index in tqdm(lens_input.index, total=len(lens_input.index), disable=not verbose):
            collector.collect(
                method.project(method.extract(lens_input, single_index.source), single_index.source),
                single_index.output,
            )

        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, lens_input: LensInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        """Describe the run ``apply`` would make, without a forward pass and without touching ``save_dir``.

        The input's index is resolved exactly as the run resolves it, and kept for ``apply``.

        Args:
            lens_input: The config and prompt the run would use.
            save_dir: The directory the run would write into.
            verbose: Whether to also log the plan.

        Returns:
            The plan: one step per index row and one forward pass, or none for a lens that reads weights.
        """
        method = LensMethodFactory.create(self.model, lens_input.config)
        lens_input.prepare_index(self.model)
        reads_activations = method.reads_activations

        plan = OperationPlan(
            operation="Lens",
            method=lens_input.config.method.type,
            config=lens_input.config,
            steps=len(lens_input.index),
            steps_label="element",
            forward_passes=1 if reads_activations else 0,
            index=[lens_input.index.to_df()],
            prompts=[("prompt", lens_input.prompt, lens_input.tokens.shape[1])] if reads_activations else [],
            activations=sorted(lens_input.index.source_activation_names()) if reads_activations else [],
            save_dir=save_dir,
            outputs=self.collector_class(lens_input, self.model, save_dir).output_files(),
        )

        if verbose:
            logger.info("%s", plan)
        return plan
