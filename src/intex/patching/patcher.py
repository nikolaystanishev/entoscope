import torch
from tqdm import tqdm

from intex.model import Model
from intex.operation.collector import AbstractCollector
from intex.operation.operator import Operator
from intex.operation.plan import OperationPlan
from intex.patching.collector import PatchingCollector
from intex.patching.input import PatchingInput
from intex.patching.metric import AbstractMetric
from intex.patching.processor import HookedPatchProcessor
from intex.util.collections import batch_generator, zip_fill


class Patcher(Operator):
    def __init__(
        self,
        model: Model,
        collector_class: type[AbstractCollector] = PatchingCollector,
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
    ) -> AbstractCollector | None:
        collector = self._patch(patching_input, save_dir, verbose)
        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, patching_input: PatchingInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        patching_input.prepare_index(self.model)

        runs = len(list(zip_fill(*patching_input.index))) if patching_input.index else 0
        batches = len(list(batch_generator(list(range(runs)), self.batch_size)))

        plan = OperationPlan(
            operation="Patcher",
            config=patching_input.config,
            steps=runs,
            steps_label="patch run",
            # One patched run per batch, on top of the two cached runs the input prepares.
            forward_passes=batches + 2,
            batches=batches,
            index=[idx.to_df() for idx in patching_input.index],
            prompts=[
                ("source", patching_input.source_prompt, patching_input.source_tokens.shape[1]),
                ("target", patching_input.target_prompt, patching_input.target_tokens.shape[1]),
            ],
            activations=patching_input.activation_names(),
            save_dir=save_dir,
            outputs=self.collector_class(patching_input, save_dir).output_files(),
        )

        if verbose:
            print(plan)
        return plan

    def _patch(self, patching_input: PatchingInput, save_dir: str | None, verbose: bool = True) -> AbstractCollector:
        patching_input.prepare(self.model)
        collector = self.collector_class(patching_input, save_dir)
        patching_index = patching_input.index
        collector.prepare()
        patching_index = list(zip_fill(*patching_index))

        for index in tqdm(
            batch_generator(patching_index, self.batch_size),
            total=(len(patching_index) + self.batch_size - 1) // self.batch_size,
            disable=not verbose,
        ):
            fwd_hooks = [
                HookedPatchProcessor.hook(idx.set_batch_index(i), patching_input)
                for i, group in enumerate(index)
                for idx in group
            ]
            patched_logits = self.model.run_with_hooks(
                torch.cat([patching_input.target_tokens] * len(index), dim=0), fwd_hooks=fwd_hooks
            ).detach()[:, -1, :]

            for idx, logits in zip(index, patched_logits, strict=False):
                metric_value = AbstractMetric.get(patching_input.config.metric).compute(patching_input, logits).item()
                collector.collect(logits, idx[0].output, metric_value)

        return collector
