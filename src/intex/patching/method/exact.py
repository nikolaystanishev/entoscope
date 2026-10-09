from collections.abc import Iterator

import torch
from tqdm import tqdm

from intex.patching.input import PatchingInput
from intex.patching.method.patching import AbstractPatchingMethod, RowResult
from intex.patching.metric import AbstractMetric
from intex.patching.processor import HookedPatchProcessor
from intex.util.collections import batch_generator, zip_fill


class ExactPatchingMethod(AbstractPatchingMethod):
    """Patch and run the model, once per index row.

    The reference the other methods approximate: nothing is assumed about how the model responds
    to the intervention, at the price of a forward pass per row.
    """

    batched = True

    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        rows = list(zip_fill(*patching_input.index))

        for batch in tqdm(
            batch_generator(rows, self.batch_size),
            total=(len(rows) + self.batch_size - 1) // self.batch_size,
            disable=not verbose,
        ):
            fwd_hooks = [
                HookedPatchProcessor.hook(idx, patching_input, row) for row, group in enumerate(batch) for idx in group
            ]
            patched_logits = self.model.run_with_hooks(
                torch.cat([patching_input.target_tokens] * len(batch), dim=0), fwd_hooks=fwd_hooks
            ).detach()[:, -1, :]
            # One transfer for the whole batch rather than one per row.
            generations = patched_logits.argmax(dim=-1).tolist()

            for group, logits, generation in zip(batch, patched_logits, generations, strict=True):
                yield RowResult(
                    group[0].output,
                    self._report(
                        patching_input,
                        {
                            metric: AbstractMetric.get(metric).compute(patching_input, logits)
                            - patching_input.target_baseline(metric)
                            for metric in self.config.metric
                        },
                    ),
                    generation=generation,
                )

    def passes(self, runs: int, batches: int) -> tuple[int, int]:
        return batches, 0
