import os
from typing import Optional

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from intex.operation.operator import Operator
from intex.patching.collector import AbstractCollector
from intex.patching.index import SinglePatchingIndex
from intex.patching.input import PatchingInput
from intex.patching.metric import AbstractMetric
from intex.patching.setter import HookedPatchSetter
from intex.util.collections import batch_generator, flatten, to_dict, zip_fill
from intex.util.data import create_if_not_exists, write_json
from intex.util.executor import none_executor


class Patcher(Operator):
    def __init__(
        self,
        model: HookedTransformer,
        patching_metric: AbstractMetric | None = None,
        patching_collector_factory: type[AbstractCollector] | None = None,
        batch_size: int = 1,
    ):
        self.model = model
        self.patching_metric = patching_metric
        self.patching_collector_factory = patching_collector_factory
        self.batch_size = batch_size

    def apply(
        self,
        patching_input: list[PatchingInput],
        save_dir: Optional[list[str]] = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> Optional[list[AbstractCollector]]:
        if save_dir is None:
            save_dir = [None] * len(patching_input)

        patching_elements = []
        save_elements = []
        for el, sd in zip(patching_input, save_dir):
            collector = self.patching_collector_factory()
            patching_index = el.get_index(self.model)
            setters = HookedPatchSetter()
            save_elements.append((el, collector, sd))
            collector.prepare(patching_index[0])
            for index in zip_fill(*patching_index):
                patching_elements.append((el, index, setters, collector, sd))

        self._patch(patching_elements, verbose)
        self._save(save_elements)

        if return_collector:
            return list(map(lambda x: x[1], save_elements))
        return None

    def _patch(
        self,
        patching_elements: list[
            tuple[PatchingInput, SinglePatchingIndex, list[HookedPatchSetter], AbstractCollector, str]
        ],
        verbose: bool = True,
    ):
        for patching_elements_batch in tqdm(
            batch_generator(patching_elements, self.batch_size),
            total=len(patching_elements) // self.batch_size + 1,
            disable=not verbose,
        ):
            input_batch, index_batch, setters_batch, collector_batch, save_dir_batch = zip(*patching_elements_batch)

            fwd_hooks = flatten(
                [
                    [setter.hook(idx.set_batch_index(i), patching_input) for idx, setter in zip(index, setters)]
                    for i, (patching_input, index, setters) in enumerate(zip(input_batch, index_batch, setters_batch))
                ]
            )

            logits_index = [patching_input.target_tokens.shape[1] - 1 for patching_input in input_batch]
            patched_logits = self.model.run_with_hooks(
                [patching_input.target_prompt for patching_input in input_batch], fwd_hooks=fwd_hooks
            ).detach()[range(len(logits_index)), logits_index, :]
            for patching_input, index, collector, logits, save_dir in zip(
                input_batch, index_batch, collector_batch, patched_logits, save_dir_batch
            ):
                metric_value = (
                    self.patching_metric(patching_input.metric_options, logits.unsqueeze(0))
                    if self.patching_metric is not None
                    else None
                )
                collector.collect(
                    none_executor(lambda el: el.source_generation, patching_input.metric_options),
                    none_executor(lambda el: el.target_generation, patching_input.metric_options),
                    logits.unsqueeze(0),
                    index[0].output,
                    metric_value,
                    save_dir,
                )

    def _save(self, save_elements: list[tuple[PatchingInput, AbstractCollector, str]]):
        for patching_input, collector, save_dir in zip(*zip(*save_elements)):
            if save_dir is not None:
                create_if_not_exists(save_dir)
                torch.save(getattr(collector, "metric_output", None), os.path.join(save_dir, "activation_patching.pt"))
                collector.save(save_dir)
                patching_input.metric_options.save(save_dir)
                for i, idx in enumerate(patching_input.get_index(self.model)):
                    idx.to_csv(os.path.join(save_dir, f"activation_patching_index-{i}.csv"))
                write_json(os.path.join(save_dir, "activation_patching_config.json"), to_dict(patching_input.config))
                print(f"{patching_input.config.patching_id} done - results saved to {save_dir}")
