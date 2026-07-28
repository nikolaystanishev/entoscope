from collections.abc import Callable, Iterator

import torch
from tqdm import tqdm

from intex.patching.config import PatchingMetric
from intex.patching.input import PatchingInput
from intex.patching.method.patching import AbstractPatchingMethod
from intex.patching.metric import AbstractMetric
from intex.patching.processor import SourceStrategy
from intex.util.collections import zip_fill

Gradients = dict[PatchingMetric, dict[str, torch.Tensor]]


class AttributionPatchingMethod(AbstractPatchingMethod):
    """Attribution patching — the first-order Taylor expansion of the true indirect effect.

    ``IE ~= grad(m) . (a_patch - a_clean)``, with the gradient taken on the target run at the
    activation a patch would overwrite. One backward pass scores the whole index, so the cost is
    flat in the number of rows rather than linear in it; the price is that the estimate is only as
    good as the model's local linearity, which a large intervention (zero ablation of a whole
    layer, say) leaves well behind.

    The intervention itself is unchanged: the replacement value comes from the same
    ``SourceStrategy`` the exact method uses, so resampling, zero ablation, scaling and noise are
    all scored the same way.
    """

    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[tuple[int, dict[str, float]]]:
        self._check_available(patching_input)
        gradients = self._gradients(patching_input, verbose)
        rows = list(zip_fill(*patching_input.index))

        for group in tqdm(rows, disable=not verbose):
            deltas: dict[PatchingMetric, torch.Tensor] = {}
            for index in group:
                name = index.target.transformer_activation()
                slices = index.target.slices()
                clean = patching_input.target_cache[name][slices]
                replacement = SourceStrategy.get(index.source.config.type).extract(index.source, patching_input)
                change = replacement.to(clean) - clean
                for metric in self.config.metric:
                    # A row patching several elements at once applies them on one run, and to first
                    # order that run's effect is the sum of theirs.
                    contribution = (gradients[metric][name][slices] * change).sum()
                    deltas[metric] = deltas[metric] + contribution if metric in deltas else contribution

            yield group[0].output, self._report(patching_input, deltas)

    def passes(self, runs: int, batches: int) -> tuple[int, int]:
        steps = len(self._alphas())
        return steps, steps * len(self.config.metric)

    def _check_available(self, patching_input: PatchingInput) -> None:
        """Nothing to check: a single gradient at the target's own activation asks for no second run."""

    def _alphas(self) -> list[float]:
        """Where along the path from the source activation to the target one the gradient is read.

        1.0 is the target (clean) activation itself, which is where the Taylor expansion is centred.
        """
        return [1.0]

    def _gradients(self, patching_input: PatchingInput, verbose: bool) -> Gradients:
        """``d m / d a`` for every hooked activation, averaged over the interpolation path."""
        names = patching_input.target_activation_names()
        alphas = self._alphas()
        totals: Gradients = {metric: {} for metric in self.config.metric}

        for alpha in tqdm(alphas, disable=not verbose or len(alphas) == 1):
            captured: dict[str, torch.Tensor] = {}
            fwd_hooks = [(name, self._capture(name, alpha, patching_input, captured)) for name in names]
            logits = self.model.run_with_hooks(patching_input.target_tokens, fwd_hooks=fwd_hooks)[0, -1]

            for i, metric in enumerate(self.config.metric):
                value = AbstractMetric.get(metric).compute(patching_input, logits)
                # allow_unused: an index may name an activation that does not reach the logits at
                # all (a head's pattern the run never attends through), which is a zero gradient
                # rather than an error.
                gradients = torch.autograd.grad(
                    value,
                    list(captured.values()),
                    retain_graph=i < len(self.config.metric) - 1,
                    allow_unused=True,
                )
                for name, gradient in zip(captured, gradients, strict=True):
                    resolved = torch.zeros_like(captured[name]) if gradient is None else gradient.detach()
                    total = totals[metric].get(name)
                    totals[metric][name] = resolved if total is None else total + resolved

        return {
            metric: {name: total / len(alphas) for name, total in per_activation.items()}
            for metric, per_activation in totals.items()
        }

    def _capture(
        self, name: str, alpha: float, patching_input: PatchingInput, captured: dict[str, torch.Tensor]
    ) -> Callable:
        """A hook that moves the activation onto the interpolation point and keeps it differentiable.

        The tensor is left in the graph — detaching it would cut the path back through the layers
        below and zero their gradients — and is handed to ``autograd.grad`` as a non-leaf input.
        """

        def interpolate(activation: torch.Tensor, hook) -> torch.Tensor:
            value = (
                activation
                if alpha == 1.0
                else alpha * activation + (1 - alpha) * patching_input.source_cache[name].to(activation)
            )
            captured[name] = value
            return value

        return interpolate


class IntegratedGradientsMethod(AttributionPatchingMethod):
    """Integrated gradients in activation space (Marks et al. 2024, Eq. 4).

    Averages the gradient over ``interpolation_steps`` points on the straight path from the source
    activation to the target one instead of reading it once at the target. That recovers the
    effects a flat gradient at the clean point hides, and costs one forward and one backward pass
    per step — still flat in the size of the index.

    Every hooked activation is interpolated on the same pass, which is what keeps the cost at
    ``interpolation_steps`` passes rather than that many per row.

    The path runs between the two runs' activations, so it is the path the intervention actually
    traverses only when the source resamples from the source run. A value-based source (zero
    ablation, scaling, noise) writes something off that path, and the averaged gradient then
    belongs to a change other than the one being scored — attribution patching, whose expansion
    point does not depend on a path at all, is the estimate to reach for there.
    """

    def _check_available(self, patching_input: PatchingInput) -> None:
        if patching_input.source_prompt is None:
            raise ValueError(
                "integrated_gradients interpolates towards the source run's activations, which needs a source prompt"
            )
        if patching_input.source_tokens.shape[1] != patching_input.target_tokens.shape[1]:
            raise ValueError(
                "integrated_gradients interpolates the two runs' activations position by position, so the prompts "
                f"must tokenize to the same length (source {patching_input.source_tokens.shape[1]}, "
                f"target {patching_input.target_tokens.shape[1]})"
            )

    def _alphas(self) -> list[float]:
        steps = self.config.interpolation_steps
        return [(step + 1) / steps for step in range(steps)]
