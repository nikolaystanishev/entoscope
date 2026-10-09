from intex.model import Model
from intex.operation.operator import Operator
from intex.operation.plan import OperationPlan
from intex.patching.collector import PatchingCollector, PatchingVisualizationCollector
from intex.patching.input import PatchingInput
from intex.patching.method.factory import PatchingMethodFactory
from intex.util.collections import batch_generator, zip_fill


class Patcher(Operator):
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
        collector = self._patch(patching_input, save_dir, verbose)
        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, patching_input: PatchingInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        # Built as the run builds it, so a plan fails wherever its run would.
        method = PatchingMethodFactory.create(patching_input.config, self.model, self.batch_size)
        patching_input.prepare_index(self.model)

        runs = len(list(zip_fill(*patching_input.index))) if patching_input.index else 0
        batches = len(list(batch_generator(list(range(runs)), self.batch_size)))
        forward_passes, backward_passes = method.passes(runs, batches)

        prompts = [("target", patching_input.target_prompt, patching_input.target_tokens.shape[1])]
        if patching_input.source_prompt is not None:
            prompts.insert(0, ("source", patching_input.source_prompt, patching_input.source_tokens.shape[1]))

        plan = OperationPlan(
            operation="Patcher",
            method=patching_input.config.method,
            config=patching_input.config,
            steps=runs,
            steps_label="patch run",
            # The method's own passes, on top of the cached run the input prepares per prompt.
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
            print(plan)
        return plan

    def _patch(self, patching_input: PatchingInput, save_dir: str | None, verbose: bool = True) -> PatchingCollector:
        # Built before the cached runs, so a method that cannot be built fails before the run does any work.
        method = PatchingMethodFactory.create(patching_input.config, self.model, self.batch_size)
        patching_input.prepare(self.model)
        collector = self.collector_class(patching_input, self.model, save_dir)
        collector.prepare()

        for row in method.scores(patching_input, verbose):
            collector.collect(row)

        return collector
