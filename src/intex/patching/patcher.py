import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from intex.operation.collector import AbstractCollector
from intex.operation.operator import Operator
from intex.patching.collector import DefaultCollector
from intex.patching.input import PatchingInput
from intex.patching.metric import AbstractMetric
from intex.patching.processor import HookedPatchProcessor
from intex.util.collections import batch_generator, zip_fill


class Patcher(Operator):
    def __init__(
        self,
        model: HookedTransformer,
        collector_class: type[AbstractCollector] = DefaultCollector,
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
