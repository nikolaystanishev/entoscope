from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from tqdm import tqdm

from entoscope.patching.config import PatchingMethodConfig, PatchingMetric
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.patching import AbstractPatchingMethod, PatchingMethodConfigT, RowResult
from entoscope.patching.metric import AbstractMetric
from entoscope.patching.processor import SourceStrategy

Gradients = dict[PatchingMetric, dict[str, torch.Tensor]]


class AttributionPatchingMethod(AbstractPatchingMethod[PatchingMethodConfigT]):
    """Attribution patching - the first-order Taylor expansion of the true indirect effect.

    ``IE ~= grad(m) . (a_patch - a_clean)``, with the gradient taken on the target run at the
    activation a patch would overwrite. One backward pass scores the whole index, so the cost is
    flat in the number of rows rather than linear in it; the price is that the estimate is only as
    good as the model's local linearity, which a large intervention (zero ablation of a whole
    layer, say) leaves well behind.

    The intervention itself is unchanged: the replacement value comes from the same
    ``SourceStrategy`` the exact method uses, so resampling, zero ablation, scaling and noise are
    all scored the same way.
    """

    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        self._check_available(patching_input)
        gradients = self._gradients(patching_input, verbose)

        for output, group in enumerate(tqdm(patching_input.rows(), disable=not verbose)):
            deltas: dict[PatchingMetric, torch.Tensor] = {}
            for index in group:
                name = index.target.transformer_activation()
                slices = index.target.slices()
                clean = patching_input.target_cache[name][slices]
                replacement = SourceStrategy.get(index.source.config.type).extract(index.source, patching_input)
                change = replacement.to(clean) - clean
                for metric in self.config.metric:
                    contribution = (gradients[metric][name][slices] * change).sum()
                    deltas[metric] = deltas[metric] + contribution if metric in deltas else contribution

            yield RowResult(output, self._report(patching_input, deltas))

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

    @torch.enable_grad()
    def _gradients(self, patching_input: PatchingInput, verbose: bool) -> Gradients:
        """``d m / d a`` for every hooked activation, averaged over the interpolation path.

        Taken with autograd on whatever the caller has set, so a run under ``torch.no_grad()`` scores the same.
        """
        names = patching_input.target_activation_names()
        alphas = self._alphas()
        totals: Gradients = {metric: {} for metric in self.config.metric}

        for alpha in tqdm(alphas, disable=not verbose or len(alphas) == 1):
            captured: dict[str, torch.Tensor] = {}
            fwd_hooks = [(name, self._capture(name, alpha, patching_input, captured)) for name in names]
            logits = self.model.run_with_hooks(patching_input.target_tokens, fwd_hooks=fwd_hooks)[0, -1]

            for i, metric in enumerate(self.config.metric):
                value = AbstractMetric.get(metric).compute(patching_input, logits)
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

        The tensor is left in the graph - detaching it would cut the path back through the layers
        below and zero their gradients - and is handed to ``autograd.grad`` as a non-leaf input. One
        that no parameter reaches, as on a frozen model, has no graph to cut and becomes a leaf instead.
        """

        def interpolate(activation: torch.Tensor, hook) -> torch.Tensor:
            value = (
                activation
                if alpha == 1.0
                else alpha * activation + (1 - alpha) * patching_input.source_cache[name].to(activation)
            )
            if not value.requires_grad:
                value = value.detach().requires_grad_()
            captured[name] = value
            return value

        return interpolate


@dataclass(kw_only=True)
class IntegratedGradientsConfig(PatchingMethodConfig):
    """``interpolation_steps`` is the number of points the gradient is averaged over."""

    interpolation_steps: int = 10

    def __post_init__(self):
        if self.interpolation_steps < 1:
            raise ValueError(f"interpolation_steps must be at least 1, got {self.interpolation_steps}")


class IntegratedGradientsMethod(AttributionPatchingMethod[IntegratedGradientsConfig]):
    """Integrated gradients in activation space (Marks et al. 2024, Eq. 4).

    Averages the gradient over ``interpolation_steps`` points on the straight path from the source
    activation to the target one instead of reading it once at the target. That recovers the
    effects a flat gradient at the clean point hides, and costs one forward and one backward pass
    per step - still flat in the size of the index.

    Every hooked activation is interpolated on the same pass, which is what keeps the cost at
    ``interpolation_steps`` passes rather than that many per row.

    The path runs between the two runs' activations, so it is the path the intervention actually
    traverses only when the source resamples from the source run. A value-based source (zero
    ablation, scaling, noise) writes something off that path, and the averaged gradient then
    belongs to a change other than the one being scored - attribution patching, whose expansion
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
        steps = self.method_config.interpolation_steps
        return [(step + 1) / steps for step in range(steps)]
