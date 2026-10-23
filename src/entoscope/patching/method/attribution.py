from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from tqdm import tqdm

from entoscope.model.activation import writable
from entoscope.patching.config import PatchingMethodConfig
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.patching import AbstractPatchingMethod, PatchingMethodConfigT, RowResult
from entoscope.patching.processor import SourceStrategy

# Keyed by the metric's name, then by the transformer activation's.
Gradients = dict[str, dict[str, torch.Tensor]]


class AttributionPatchingMethod(AbstractPatchingMethod[PatchingMethodConfigT]):
    """Attribution patching, the first-order Taylor expansion of the true indirect effect.

    ``IE ~= grad(m) . (a_patch - a_clean)``, with the gradient taken on the target run at the
    activation a patch would overwrite. One backward pass scores the whole index, so the cost is
    flat in the number of rows rather than linear in it. The price is that the estimate is only as
    good as the model's local linearity, which a large intervention (zero ablation of a whole
    layer, say) leaves well behind.

    The intervention itself is unchanged: the replacement value comes from the same
    ``SourceStrategy`` the exact method uses, so resampling, zero ablation, scaling and noise are
    all scored the same way.
    """

    def scores(self, patching_input: PatchingInput, verbose: bool = True) -> Iterator[RowResult]:
        gradients = self._gradients(patching_input, verbose)

        for output, group in enumerate(tqdm(patching_input.rows(), disable=not verbose)):
            deltas: dict[str, torch.Tensor] = {}
            for index in group:
                name = index.target.transformer_activation()
                clean = index.target.read_slices(patching_input.target_cache[name])
                replacement = SourceStrategy.get(index.source.config.type).extract(index.source, patching_input)
                change = replacement.to(clean) - clean
                for metric in self.config.metric:
                    contribution = (index.target.read_slices(gradients[metric][name]) * change).sum()
                    deltas[metric] = deltas[metric] + contribution if metric in deltas else contribution

            yield RowResult(output, self._report(patching_input, deltas))

    def passes(self, patching_input: PatchingInput, batches: int) -> tuple[int, int]:
        forward = len(self._passes(patching_input.target_activation_names()))
        return forward, forward * len(self.config.metric)

    def _alphas(self) -> list[float]:
        """Choose where along the path from the source activation to the target one the gradient is read.

        Returns:
            The interpolation points, 1.0 being the target (clean) activation itself, where the Taylor
            expansion is centred.
        """
        return [1.0]

    def _passes(self, names: list[str]) -> list[tuple[float, list[str]]]:
        """Pair every interpolation point with the transformer activations each pass at it interpolates.

        At the clean point a hook leaves its activation as it is, so one pass reads the gradient at every
        activation, each the total derivative through the layers above it. Anywhere else an activation is interpolated
        on a pass of its own, as Marks et al. do: on a shared pass every later hook would scale the
        gradient reaching an earlier one by alpha and move that one off its own path, so a row's score
        would depend on what else the index holds.

        Args:
            names: The transformer activation names a patch writes.

        Returns:
            One ``(alpha, names)`` per forward pass, every name appearing once per alpha.
        """
        return [
            (alpha, group)
            for alpha in self._alphas()
            for group in ([names] if alpha == 1.0 else [[name] for name in names])
        ]

    @torch.enable_grad()
    def _gradients(self, patching_input: PatchingInput, verbose: bool) -> Gradients:
        """Take ``d m / d a`` for every patched activation, averaged over the interpolation path.

        Autograd is on whatever the caller has set, so a run under ``torch.no_grad()`` scores the same.

        Args:
            patching_input: The prepared input.
            verbose: Whether to show a progress bar over the passes.

        Returns:
            Each metric's gradient with respect to every patched activation, keyed by the metric's name and
            the transformer activation's.
        """
        passes = self._passes(patching_input.target_activation_names())
        totals: Gradients = {metric: {} for metric in self.config.metric}

        for alpha, names in tqdm(passes, disable=not verbose or len(passes) == 1):
            captured: dict[str, torch.Tensor] = {}
            fwd_hooks = [(name, self._capture(name, alpha, patching_input, captured)) for name in names]
            logits = self.model.run_with_hooks(patching_input.target_tokens, fwd_hooks=fwd_hooks)[0, -1]

            for i, metric in enumerate(self.config.metric):
                gradients = torch.autograd.grad(
                    patching_input.metrics[metric].compute(patching_input, logits),
                    list(captured.values()),
                    retain_graph=i < len(self.config.metric) - 1,
                    allow_unused=True,
                )
                for name, gradient in zip(captured, gradients, strict=True):
                    resolved = torch.zeros_like(captured[name]) if gradient is None else gradient.detach()
                    total = totals[metric].get(name)
                    totals[metric][name] = resolved if total is None else total + resolved

        steps = len(self._alphas())
        return {
            metric: {name: total / steps for name, total in per_activation.items()}
            for metric, per_activation in totals.items()
        }

    def _capture(
        self, name: str, alpha: float, patching_input: PatchingInput, captured: dict[str, torch.Tensor]
    ) -> Callable:
        """Build a hook that moves an activation onto the interpolation point and keeps it differentiable.

        The tensor is left in the graph, since detaching it would cut the path back through the layers
        below and zero their gradients, and is handed to ``autograd.grad`` as a non-leaf input. One that
        no parameter reaches, as on a frozen model, has no graph to cut and becomes a leaf instead. At the
        clean point the hook hands on the tensor ``writable`` picks, as a patch would write it, so a sublayer's
        input is differentiated through that sublayer alone, as it is at every other point.

        Args:
            name: The transformer activation name.
            alpha: The interpolation point, 1.0 for the target activation.
            patching_input: The prepared input.
            captured: Where the hook stores the activation it hands on, keyed by transformer activation name.

        Returns:
            The hook function.
        """

        def interpolate(activation: torch.Tensor, hook) -> torch.Tensor:
            value = (
                writable(name, activation)
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
    """The method config of ``integrated_gradients``.

    Attributes:
        interpolation_steps: The number of points the gradient is averaged over.

    Raises:
        ValueError: If ``interpolation_steps`` is below 1.
    """

    interpolation_steps: int = 10

    def __post_init__(self):
        if self.interpolation_steps < 1:
            raise ValueError(f"interpolation_steps must be at least 1, got {self.interpolation_steps}")


class IntegratedGradientsMethod(AttributionPatchingMethod[IntegratedGradientsConfig]):
    """Integrated gradients in activation space (Marks et al. 2024, Eq. 4).

    Averages the gradient over ``interpolation_steps`` points on the straight path from the source
    activation to the target one instead of reading it once at the target. That recovers the
    effects a flat gradient at the clean point hides, at a forward and a backward pass per step for
    each transformer activation, still flat in the rows of the index.

    Each patched activation is interpolated on passes of its own, one per step but the clean one, which
    a single pass serves for all of them. The cost so grows with the number of transformer activations the
    index patches, such as the layers it sweeps, and never with its rows: every position, head or neuron of
    one is scored from the same passes.

    The path runs between the two runs' activations, so it is the path the intervention actually
    traverses only when the source resamples from the source run. A value-based source (zero
    ablation, scaling, noise) writes something off that path, and the averaged gradient then
    belongs to a change other than the one being scored. Attribution patching, whose expansion
    point does not depend on a path at all, is the estimate to reach for there.
    """

    def check(self, patching_input: PatchingInput) -> None:
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
