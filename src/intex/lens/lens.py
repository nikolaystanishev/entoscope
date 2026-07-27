from tqdm import tqdm

from intex.lens.collector import LensCollector, LensVisualizationCollector
from intex.lens.input import LensInput
from intex.lens.method.factory import LensMethodFactory
from intex.model import Model
from intex.operation.operator import Operator
from intex.operation.plan import OperationPlan


class Lens(Operator):
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
        lens_input.prepare(self.model)
        collector = self.collector_class(lens_input, self.model, save_dir)
        collector.prepare()

        for single_index in tqdm(lens_input.index, total=len(lens_input.index), disable=not verbose):
            method = LensMethodFactory.create(lens_input.config, self.model, single_index.source.activation())
            collector.collect(
                method.project(method.extract(lens_input, single_index.source), single_index.source),
                single_index.output,
            )

        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, lens_input: LensInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        lens_input.prepare_index(self.model)

        # The parameter lens reads the weights, so it neither caches activations nor runs the model.
        reads_activations = lens_input.prompt is not None

        plan = OperationPlan(
            operation="Lens",
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
            print(plan)
        return plan
