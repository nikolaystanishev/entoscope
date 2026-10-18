from collections.abc import Iterator

import torch
from tqdm import tqdm

from entoscope.patching.input import PatchingInput
from entoscope.patching.method.patching import AbstractPatchingMethod, PatchingMethodConfigT, RowResult
from entoscope.patching.metric import AbstractMetric
from entoscope.patching.processor import HookedPatchProcessor
from entoscope.util.collections import batch_generator


class ExactPatchingMethod(AbstractPatchingMethod[PatchingMethodConfigT]):
    """Patches and runs the model, once per index row.

    The reference the other methods approximate: nothing is assumed about how the model responds to the
    intervention, at the price of a forward pass per row. Its last batch is padded to the full width, so
    every patched run has the shape the cached runs were made at.
    """

    batched = True

    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        rows = patching_input.rows()
        batch_width = patching_input.batch_width

        for number, batch in enumerate(
            tqdm(
                batch_generator(rows, batch_width),
                total=(len(rows) + batch_width - 1) // batch_width,
                disable=not verbose,
            )
        ):
            fwd_hooks = [
                HookedPatchProcessor.hook(idx, patching_input, row) for row, group in enumerate(batch) for idx in group
            ]
            with torch.no_grad():
                patched_logits = self.model.run_with_hooks(
                    torch.cat([patching_input.target_tokens] * batch_width, dim=0), fwd_hooks=fwd_hooks
                )[: len(batch), -1, :]
            generations = patched_logits.argmax(dim=-1).tolist()

            for row, (logits, generation) in enumerate(zip(patched_logits, generations, strict=True)):
                yield RowResult(
                    number * batch_width + row,
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
