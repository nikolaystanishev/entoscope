import json
from dataclasses import replace
from html import escape
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from entoscope.model import Model
from entoscope.model.cache import CachedModel
from entoscope.model.config import Filter, Group, Incremental, ModelAxis, ModelLayerLocation
from entoscope.operation.config import ConfigFactory
from entoscope.patching.collector import PatchingCollector
from entoscope.patching.config import (
    MetricComparison,
    PatchingConfig,
    PatchingElementConfig,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
    metric_column,
)
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.attribution import IntegratedGradientsConfig
from entoscope.patching.patcher import Patcher

from ..conftest import (
    RenderedFigure,
    build_small_model,
    forbid_model_runs,
    record_runs,
    rendered_figure,
    rendered_figures,
)

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"
EXACT = PatchingMethodConfig(type="exact")


def _metric(
    collector: PatchingCollector, metric: PatchingMetric | None = None, comparison: MetricComparison | None = None
) -> torch.Tensor:
    """One of the result frame's metric columns, in index order - one value per patched run.

    Names nothing: the first column when neither argument is given, and a metric's own first
    comparison when only the metric is.
    """
    config = collector.operation_input.config
    if metric is None:
        column = config.metric_columns()[0]
    else:
        column = metric_column(metric, comparison if comparison is not None else config.metric[metric][0])
    return torch.tensor(collector.result[column].to_numpy(), dtype=torch.float32)


def _cell_columns(collector: PatchingCollector) -> list[str]:
    """The result's columns past the index's own: what each patched run reported, in order."""
    return [column for column in collector.result.columns if not column.startswith(("source_", "target_"))]


# What the exact method records about the patched run besides its metrics.
GENERATION_COLUMNS = ["generation", "str_generation"]


def _run(model: Model, config: PatchingConfig, **answers) -> torch.Tensor:
    return _metric(_collect(model, config, **answers))


def _collect(model: Model, config: PatchingConfig, source: str | None = SOURCE_PROMPT, **answers) -> PatchingCollector:
    collector = Patcher(model).apply(
        PatchingInput(config, source, TARGET_PROMPT, **answers), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector


def _generations(model: Model) -> tuple[int, int]:
    """The tokens the two prompts predict - the pair every logit-diff metric falls back to."""
    with torch.no_grad():
        return int(model(SOURCE_PROMPT)[0, -1].argmax()), int(model(TARGET_PROMPT)[0, -1].argmax())


def _source_activation(model: Model, name: str) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(SOURCE_PROMPT), names_filter=[name])
    return cache[name]


def _final_logits(model: Model, prompt: str) -> torch.Tensor:
    with torch.no_grad():
        return model(prompt)[0, -1]


def _unpatched_metrics(model: Model, prompt: str) -> dict[PatchingMetric, torch.Tensor]:
    """Every metric of one prompt's own run, by its formula, scoring the tokens the two prompts predict."""
    correct, distractor = _generations(model)
    logits = _final_logits(model, prompt)
    log_probabilities = logits.log_softmax(dim=-1)
    return {
        PatchingMetric.LOGIT_DIFF: logits[correct] - logits[distractor],
        PatchingMetric.PROBABILITY: logits.softmax(dim=-1)[correct],
        PatchingMetric.LOG_PROB: log_probabilities[correct],
        PatchingMetric.LOGPROB_DIFF: log_probabilities[correct] - log_probabilities[distractor],
        PatchingMetric.KL_DIVERGENCE: torch.nn.functional.kl_div(
            log_probabilities, _final_logits(model, SOURCE_PROMPT).softmax(dim=-1), reduction="sum"
        ),
    }


def _expected_logit_diff(model: Model, fwd_hooks: list, answers: tuple[int, int] | None = None) -> torch.Tensor:
    """The metric a patched run must produce, from running the same patch by hand.

    The hooks say what the patch writes and where; the logit diff is then read straight off the
    patched run, so nothing about the expected value is frozen into the test. It compares ``answers``,
    a pair of token ids, or else the tokens the two prompts predict.
    """
    correct, distractor = answers if answers is not None else _generations(model)
    with torch.no_grad():
        patched = model.run_with_hooks(model.to_tokens(TARGET_PROMPT), fwd_hooks=fwd_hooks)
    return patched[0, -1, correct] - patched[0, -1, distractor]


def _copying_hooks(model: Model, writes: list[tuple[str, tuple]]) -> list:
    """Hooks copying the source run's activation into the target run at each ``(hook name, index)``."""

    def copying(source: torch.Tensor, at: tuple):
        def write(activation: torch.Tensor, hook) -> torch.Tensor:
            activation[at] = source[at]
            return activation

        return write

    return [(name, copying(_source_activation(model, name), (slice(None), *index))) for name, index in writes]


def _last_layer_resid_element() -> PatchingElementConfig:
    return PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: -1}]),
        type=PatchingType.REPLACE,
    )


def _last_layer_resid_patch_config() -> PatchingConfig:
    """Replace the last layer's full residual stream (all positions) source -> target."""
    element = _last_layer_resid_element()
    return PatchingConfig(
        config_id="last_layer_resid_patch",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )


def _last_layer_resid_zero_ablation_config() -> PatchingConfig:
    """Zero-ablate the last layer's full residual stream (all positions): set every value to 0."""
    source = PatchingElementConfig(type=PatchingType.VALUE_REPLACE)  # produces zeros
    target = _last_layer_resid_element()  # writes them into the activation
    return PatchingConfig(
        config_id="last_layer_resid_zero_ablation",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def _resid_value_scale_config(scale: float, layer: int) -> PatchingConfig:
    """Scale a layer's source residual stream (all positions) by ``scale``, then write it into the target.

    The scaled layer is intentionally not the final one: ``resid.post`` of the last layer feeds straight
    into ``ln_final``, whose normalization cancels any uniform scaling, so the scale would have no effect
    on the logits. At an intermediate layer the scaled residual is carried forward through the skip
    connections, so the scale genuinely changes the output.
    """
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layer}]),
        type=PatchingType.VALUE_SCALE,
        value=scale,
    )
    target = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="resid_value_scale",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def _mlp_mhsa_pattern_elements(layer: int, head: int) -> list[PatchingElementConfig]:
    """Three components of one layer, each replaced from the source run:

    - the MLP post-activation neurons (``mlp.post``),
    - the multi-head self-attention queries (``attn.q``), and
    - a single attention head's pattern (``attn.pattern`` filtered to ``head``).
    """
    mlp = PatchingElementConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    mhsa = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_Q,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    pattern = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelAxis.LAYER, ModelAxis.HEAD],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layer, ModelAxis.HEAD: head}]),
        type=PatchingType.REPLACE,
    )
    return [mlp, mhsa, pattern]


def _patch_config(*elements: PatchingElementConfig, config_id: str = "patch") -> PatchingConfig:
    """Patches sharing one config are applied together, on a single forward pass."""
    return PatchingConfig(
        config_id=config_id,
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element) for element in elements],
    )


def test_activation_patching_from_source_into_target(model: Model) -> None:
    torch.manual_seed(42)

    # A full residual replacement requires matching sequence lengths between the two prompts.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_zero_ablation_patching(model: Model) -> None:
    torch.manual_seed(42)

    # Zero ablation writes zeros into the target run, so the source run's activations are never read;
    # its forward pass only supplies the source generation token that the logit-diff metric compares
    # against the target one. Distinct prompts keep those two tokens distinct.
    patching_input = PatchingInput(_last_layer_resid_zero_ablation_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    # The same ablation done by hand: the last layer's residual is zeroed on the target run.
    name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    expected = _expected_logit_diff(model, [(name, lambda activation, hook: torch.zeros_like(activation))])
    assert torch.allclose(result.squeeze(), expected, atol=1e-5)


# Intermediate layer whose residual stream is scaled; see ``_resid_value_scale_config`` for why it is
# not the final layer.
_VALUE_SCALE_LAYER = 6


def _run_value_scale(model: Model, scale: float) -> torch.Tensor:
    # Scaling the source residual and writing it into the target requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_resid_value_scale_config(scale, _VALUE_SCALE_LAYER), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (a single layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)
    return result.squeeze()


def test_value_scale_patching(model: Model) -> None:
    torch.manual_seed(42)

    scaled = _run_value_scale(model, scale=2.0)
    unscaled = _run_value_scale(model, scale=1.0)

    # Each run is checked against the same scaling done by hand: the source's residual, multiplied,
    # written over the target's at that layer.
    name = f"blocks.{_VALUE_SCALE_LAYER}.hook_resid_post"
    source_residual = _source_activation(model, name)
    for result, scale in ((scaled, 2.0), (unscaled, 1.0)):
        expected = _expected_logit_diff(model, [(name, lambda activation, hook, s=scale: s * source_residual)])
        assert torch.allclose(result, expected, atol=1e-5)

    # Scaling by 2x is not the same as writing the residual through unchanged, so the scale takes effect.
    assert not torch.allclose(scaled, unscaled)


# Layer and head whose MLP, attention output, and head pattern are patched together below.
_MULTI_PATCH_LAYER = 5
_MULTI_PATCH_HEAD = 7


def _multi_patch_hooks(model: Model) -> list:
    """The three replacements done by hand, from the source run's own activations."""
    layer, head = _MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD
    post, queries, pattern = (
        f"blocks.{layer}.mlp.hook_post",
        f"blocks.{layer}.attn.hook_q",
        f"blocks.{layer}.attn.hook_pattern",
    )
    # Read from the source run up front: a forward pass inside a hook would run the hooks again.
    source_post, source_queries, source_pattern = (_source_activation(model, name) for name in (post, queries, pattern))

    def patch_one_head(activation: torch.Tensor, hook) -> torch.Tensor:
        patched = activation.clone()
        patched[:, head] = source_pattern[:, head]
        return patched

    return [
        (post, lambda activation, hook: source_post),
        (queries, lambda activation, hook: source_queries),
        (pattern, patch_one_head),
    ]


def test_multi_config_patching_mlp_mhsa_and_pattern(model: Model) -> None:
    torch.manual_seed(42)

    # Replacing full activations (and a head's pattern) from source requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    elements = _mlp_mhsa_pattern_elements(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD)
    patching_input = PatchingInput(_patch_config(*elements), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # The config carries three patches; each becomes its own index, targeting -- in order -- the MLP
    # post-activation, the MHSA queries, and the attention pattern.
    assert [idx.target_config.activation for idx in patching_input.index] == [
        ModelLayerLocation.MLP_POST,
        ModelLayerLocation.ATTENTION_Q,
        ModelLayerLocation.ATTENTION_PATTERN,
    ]

    # All three patches are applied together on a single forward pass and the collector keys the result
    # off the first patch's (single) output coordinate -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    # The three replacements done by hand, on one forward pass, give the same logit diff...
    assert torch.allclose(result.squeeze(), _expected_logit_diff(model, _multi_patch_hooks(model)), atol=1e-5)

    # ...and none of the three on its own does, so all of them are taking effect at once.
    for element in elements:
        alone = _run(model, _patch_config(element, config_id="single_patch"))
        assert not torch.allclose(result.squeeze(), alone.squeeze(), atol=1e-3)


# --------------------------------------------------------------------------------------------------
# The metrics - every scalar the formulas name, read off the patched run's final position
# --------------------------------------------------------------------------------------------------
def _no_op_element() -> PatchingElementConfig:
    """Position 0 is the prompts' shared prefix, so writing the source's activation there changes
    nothing: the patched run is the target run, and every metric is then computable from two plain
    forward passes."""
    return PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        coordinates=[{ModelAxis.LAYER: -1, ModelAxis.POSITION: 0}],
        type=PatchingType.REPLACE,
    )


def _metric_config(
    *metrics: PatchingMetric,
    comparison: MetricComparison = MetricComparison.ABSOLUTE,
    element: PatchingElementConfig | None = None,
) -> PatchingConfig:
    """Every named metric under the one comparison - the shape most of these tests want."""
    element = element if element is not None else _no_op_element()
    return PatchingConfig(
        config_id="metrics",
        metric={metric: [comparison] for metric in metrics},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )


def test_every_output_distribution_metric_matches_its_formula(model: Model) -> None:
    torch.manual_seed(42)

    # The patch writes the shared prefix, so the patched run is the target run.
    collector = _collect(model, _metric_config(*PatchingMetric))

    for metric, value in _unpatched_metrics(model, TARGET_PROMPT).items():
        assert torch.allclose(_metric(collector, metric).squeeze(), value, atol=1e-4), metric


def test_the_kl_divergence_runs_from_the_source_distribution_to_the_patched_one(model: Model) -> None:
    torch.manual_seed(42)

    # The divergence is not symmetric, so the direction is the whole content of the metric: the
    # source run is the reference the patch is trying to reproduce, and the patched run is scored
    # against it. Swapping the two gives a different number, which is what makes this observable.
    result = _run(model, _metric_config(PatchingMetric.KL_DIVERGENCE))

    source = _final_logits(model, SOURCE_PROMPT).softmax(dim=-1)
    target = _final_logits(model, TARGET_PROMPT)
    divergence = torch.nn.functional.kl_div
    assert torch.allclose(result.squeeze(), divergence(target.log_softmax(-1), source, reduction="sum"), atol=1e-4)
    assert not torch.allclose(
        result.squeeze(), divergence(source.log(), target.softmax(-1), reduction="sum"), atol=1e-3
    )


def test_a_patch_that_reproduces_the_source_run_leaves_no_divergence(model: Model) -> None:
    torch.manual_seed(42)

    # Replacing the last layer's whole residual makes the patched run the source run, so there is
    # nothing left between the two distributions.
    result = _run(model, _metric_config(PatchingMetric.KL_DIVERGENCE, element=_last_layer_resid_element()))

    assert torch.allclose(result.squeeze(), torch.tensor(0.0), atol=1e-5)


def test_the_logprob_difference_is_the_logit_difference(model: Model) -> None:
    torch.manual_seed(42)

    # The softmax's normalizer is shared by the two tokens and cancels in the subtraction. Worth
    # pinning: the papers name them separately, and a reader is entitled to know they agree.
    collector = _collect(model, _metric_config(PatchingMetric.LOGIT_DIFF, PatchingMetric.LOGPROB_DIFF))

    assert torch.allclose(
        _metric(collector, PatchingMetric.LOGIT_DIFF), _metric(collector, PatchingMetric.LOGPROB_DIFF), atol=1e-4
    )


def test_several_metrics_come_off_one_run(model: Model) -> None:
    torch.manual_seed(42)

    element = _last_layer_resid_element()
    together = _collect(model, _metric_config(*PatchingMetric, element=element))

    # One column per metric, named after it, in the order the config asked for them...
    assert _cell_columns(together) == [*(metric.value for metric in PatchingMetric), *GENERATION_COLUMNS]
    # ...each holding what the same metric produces when it is the only one asked for.
    for metric in PatchingMetric:
        alone = _run(model, _metric_config(metric, element=element))
        assert torch.allclose(_metric(together, metric), alone, atol=1e-5), metric


# --------------------------------------------------------------------------------------------------
# The comparisons - the patched run against the two it sits between
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("metric", list(PatchingMetric))
def test_the_three_comparisons_agree_on_one_patch(model: Model, metric: PatchingMetric) -> None:
    torch.manual_seed(42)

    element = _last_layer_resid_element()
    target = _unpatched_metrics(model, TARGET_PROMPT)[metric]
    source = _unpatched_metrics(model, SOURCE_PROMPT)[metric]

    absolute = _run(model, _metric_config(metric, element=element))
    effect = _run(model, _metric_config(metric, element=element, comparison=MetricComparison.INDIRECT_EFFECT))
    normalized = _run(model, _metric_config(metric, element=element, comparison=MetricComparison.NORMALIZED))

    # The indirect effect is the absolute metric with the target run subtracted -- signed so that a
    # patch moving the model towards the source scores positive, which for a divergence is a drop.
    sign = -1.0 if metric is PatchingMetric.KL_DIVERGENCE else 1.0
    assert torch.allclose(effect.squeeze(), sign * (absolute.squeeze() - target), atol=1e-4)
    # Replacing the last layer's whole residual reproduces the source run, so the normalized form
    # reads exactly 1 -- whichever metric measures it.
    assert torch.allclose(normalized.squeeze(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(normalized.squeeze(), (absolute.squeeze() - target) / (source - target), atol=1e-4)


def test_a_patch_that_changes_nothing_has_no_indirect_effect(model: Model) -> None:
    torch.manual_seed(42)

    # The doc's zero point: an ineffective patch reproduces the target run, so its effect is nil.
    result = _run(model, _metric_config(*PatchingMetric, comparison=MetricComparison.INDIRECT_EFFECT))

    assert torch.allclose(result.squeeze(), torch.tensor(0.0), atol=1e-5)


def test_each_metric_is_reported_under_its_own_comparison(model: Model) -> None:
    torch.manual_seed(42)

    # The reason the comparison sits on the metric: a divergence is read as an effect, a logit
    # difference as a fraction of the source run's own distance, and both off one patched run.
    element = _last_layer_resid_element()
    together = _collect(
        model,
        PatchingConfig(
            config_id="mixed",
            metric={
                PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED],
                PatchingMetric.KL_DIVERGENCE: [MetricComparison.INDIRECT_EFFECT],
            },
            method=EXACT,
            patch=[SinglePatchingConfig(source=element, target=element)],
        ),
    )

    assert _cell_columns(together) == ["normalized_logit_diff", "indirect_effect_kl_divergence", *GENERATION_COLUMNS]
    for metric, comparison in (
        (PatchingMetric.LOGIT_DIFF, MetricComparison.NORMALIZED),
        (PatchingMetric.KL_DIVERGENCE, MetricComparison.INDIRECT_EFFECT),
    ):
        alone = _run(model, _metric_config(metric, element=element, comparison=comparison))
        assert torch.allclose(_metric(together, metric), alone, atol=1e-5), metric


def test_one_metric_may_be_reported_under_several_comparisons(model: Model) -> None:
    torch.manual_seed(42)

    element = _last_layer_resid_element()
    comparisons = [MetricComparison.NORMALIZED, MetricComparison.ABSOLUTE, MetricComparison.INDIRECT_EFFECT]
    together = _collect(
        model,
        PatchingConfig(
            config_id="thrice",
            metric={PatchingMetric.LOGIT_DIFF: comparisons},
            method=EXACT,
            patch=[SinglePatchingConfig(source=element, target=element)],
        ),
    )

    # Three columns off one change in one metric, each agreeing with the run that asks for it alone.
    assert _cell_columns(together) == [
        "normalized_logit_diff",
        "logit_diff",
        "indirect_effect_logit_diff",
        *GENERATION_COLUMNS,
    ]
    for comparison in comparisons:
        alone = _run(model, _metric_config(PatchingMetric.LOGIT_DIFF, element=element, comparison=comparison))
        assert torch.allclose(_metric(together, PatchingMetric.LOGIT_DIFF, comparison), alone, atol=1e-5), comparison


def test_a_second_comparison_costs_no_extra_backward_pass(model: Model) -> None:
    # The gradient methods differentiate once per metric, and a comparison only re-reads that
    # gradient's delta - so naming three of them is free where naming a second metric is not.
    element = _last_layer_resid_element()

    def backward_passes(metric: dict) -> int:
        config = PatchingConfig(
            config_id="cost",
            metric=metric,
            method=PatchingMethodConfig(type="attribution"),
            patch=[SinglePatchingConfig(source=element, target=element)],
        )
        return Patcher(model).plan(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False).backward_passes

    assert backward_passes({PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]}) == 1
    assert backward_passes({PatchingMetric.LOGIT_DIFF: list(MetricComparison)}) == 1
    assert (
        backward_passes(
            dict.fromkeys((PatchingMetric.LOGIT_DIFF, PatchingMetric.PROBABILITY), [MetricComparison.ABSOLUTE])
        )
        == 2
    )


# --------------------------------------------------------------------------------------------------
# The answer tokens the difference metrics compare
# --------------------------------------------------------------------------------------------------
def test_the_answer_tokens_are_what_the_metric_compares(model: Model) -> None:
    torch.manual_seed(42)

    element = _last_layer_resid_element()
    named = _run(
        model,
        _metric_config(PatchingMetric.LOGIT_DIFF, element=element),
        correct_token=" Paris",
        distractor_token=" Madrid",
    )
    logits = _final_logits(model, SOURCE_PROMPT)  # the patch reproduces the source run
    expected = logits[model.to_single_token(" Paris")] - logits[model.to_single_token(" Madrid")]

    assert torch.allclose(named.squeeze(), expected, atol=1e-4)
    # A token id says the same thing as its string...
    by_id = _run(
        model,
        _metric_config(PatchingMetric.LOGIT_DIFF, element=element),
        correct_token=model.to_single_token(" Paris"),
        distractor_token=model.to_single_token(" Madrid"),
    )
    assert torch.equal(by_id, named)
    # ...and naming them at all is what makes the run something other than the argmax default.
    assert not torch.allclose(named, _run(model, _metric_config(PatchingMetric.LOGIT_DIFF, element=element)))


def test_two_identical_answer_tokens_raise_rather_than_scoring_zero(model: Model) -> None:
    # A difference metric whose two tokens coincide is identically zero at every location, which
    # reads as 'the patch did nothing' everywhere rather than as the misconfiguration it is.
    with pytest.raises(ValueError, match="identically zero"):
        _run(model, _last_layer_resid_patch_config(), correct_token=" Paris", distractor_token=" Paris")

    # Patching a prompt with itself is the same trap through the argmax fallback.
    with pytest.raises(ValueError, match="identically zero"):
        _collect(model, _last_layer_resid_patch_config(), source=TARGET_PROMPT)


def test_a_metric_that_needs_no_distractor_does_not_mind(model: Model) -> None:
    torch.manual_seed(42)

    # Only the difference metrics compare two tokens, so the guard above must not reach the others.
    result = _collect(
        model,
        _metric_config(PatchingMetric.PROBABILITY, PatchingMetric.KL_DIVERGENCE, element=_last_layer_resid_element()),
        source=TARGET_PROMPT,
    )

    assert len(result.result) == 1


# --------------------------------------------------------------------------------------------------
# The generation - the token each patched run predicts next
# --------------------------------------------------------------------------------------------------
def test_a_patch_that_reproduces_the_source_run_generates_what_the_source_does(model: Model) -> None:
    collector = _collect(model, _last_layer_resid_patch_config())
    source_generation, target_generation = _generations(model)

    # The two prompts predict different tokens, so this tells the patched run apart from the target run.
    assert source_generation != target_generation
    assert collector.result["generation"].tolist() == [source_generation]
    assert collector.result["str_generation"].tolist() == [model.to_single_str_token(source_generation)]


def test_a_patch_that_changes_nothing_generates_what_the_target_does(model: Model) -> None:
    collector = _collect(model, _metric_config(PatchingMetric.PROBABILITY))

    assert collector.result["generation"].tolist() == [_generations(model)[1]]


@pytest.mark.parametrize("batch_size", [1, 3])
def test_each_row_generates_what_its_own_patched_run_predicts(model: Model, batch_size: int) -> None:
    # The country token, patched layer by layer: early on it still carries the source's answer to the
    # last position, and by the last layers nothing reads it any more - so the rows differ, and a
    # batch that mixed them up would show.
    position = 4
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=Filter(inclusion_terms=[{ModelAxis.POSITION: position}]),
        type=PatchingType.REPLACE,
    )
    collector = Patcher(model, batch_size=batch_size).apply(
        PatchingInput(_metric_config(PatchingMetric.PROBABILITY, element=element), SOURCE_PROMPT, TARGET_PROMPT),
        return_collector=True,
        verbose=False,
    )
    assert collector is not None

    expected = []
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.hook_resid_post"
        source = _source_activation(model, name)

        def write(activation: torch.Tensor, hook, source: torch.Tensor = source) -> torch.Tensor:
            activation[:, position] = source[:, position]
            return activation

        with torch.no_grad():
            patched = model.run_with_hooks(model.to_tokens(TARGET_PROMPT), fwd_hooks=[(name, write)])
        expected.append(int(patched[0, -1].argmax()))

    assert len(set(expected)) > 1
    assert collector.result["generation"].tolist() == expected


@pytest.mark.parametrize("method", ["attribution", "integrated_gradients"])
def test_a_gradient_method_reports_no_generation(model: Model, method: str) -> None:
    # A gradient method never runs the patched model, so there is no next token to read off it.
    collector = _collect(model, _gradient_config(method))

    assert _cell_columns(collector) == ["indirect_effect_logit_diff"]


# --------------------------------------------------------------------------------------------------
# A run with no source prompt at all - the single-input indirect effect
# --------------------------------------------------------------------------------------------------
def test_zero_ablation_scores_its_effect_without_a_second_prompt(model: Model) -> None:
    torch.manual_seed(42)

    # Nothing is read from another run, so the intervention needs no counterfactual prompt: the
    # effect is the metric under 'do(a = 0)' minus the metric of the prompt left alone.
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]}
    patching_input = PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona")

    collector = Patcher(model, collector_class=PatchingCollector).apply(
        patching_input, return_collector=True, verbose=False
    )
    assert collector is not None

    correct, distractor = (model.to_single_token(token) for token in (" Madrid", " Barcelona"))
    name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    with torch.no_grad():
        ablated = model.run_with_hooks(
            model.to_tokens(TARGET_PROMPT),
            fwd_hooks=[(name, lambda activation, hook: torch.zeros_like(activation))],
        )[0, -1]
    clean = _final_logits(model, TARGET_PROMPT)
    expected = (ablated[correct] - ablated[distractor]) - (clean[correct] - clean[distractor])

    assert torch.allclose(_metric(collector).squeeze(), expected, atol=1e-4)


def test_a_run_without_a_source_prompt_has_to_name_its_correct_token(model: Model) -> None:
    # Without a source run there is nothing to read the correct answer off, so it has to be given.
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {PatchingMetric.PROBABILITY: [MetricComparison.INDIRECT_EFFECT]}

    with pytest.raises(ValueError, match="pass 'correct_token'"):
        Patcher(model).apply(PatchingInput(config, None, TARGET_PROMPT), verbose=False)


@pytest.mark.parametrize(
    ("metric", "comparison"),
    [
        (PatchingMetric.KL_DIVERGENCE, MetricComparison.ABSOLUTE),
        (PatchingMetric.LOGIT_DIFF, MetricComparison.NORMALIZED),
    ],
)
def test_a_metric_that_needs_the_source_run_says_so(
    model: Model, metric: PatchingMetric, comparison: MetricComparison
) -> None:
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {metric: [comparison]}

    with pytest.raises(ValueError, match="needs a source prompt"):
        Patcher(model).apply(
            PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona"),
            verbose=False,
        )


# --------------------------------------------------------------------------------------------------
# The gradient methods - the same score, estimated rather than run
# --------------------------------------------------------------------------------------------------
_GRADIENT_LAYER = 6


def _gradient_config(
    method: str | PatchingMethodConfig,
    source_type: PatchingType = PatchingType.REPLACE,
    value: float = 0.0,
    layer: int | list[int] = _GRADIENT_LAYER,
) -> PatchingConfig:
    """A method named by its type alone takes its defaults, the way a YAML file names it."""
    method = method if isinstance(method, PatchingMethodConfig) else PatchingMethodConfig.from_config(method)
    location, index_filter = [ModelAxis.LAYER], Filter(inclusion_terms=[{ModelAxis.LAYER: layer}])
    return PatchingConfig(
        config_id=method.type,
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]},
        method=method,
        patch=[
            SinglePatchingConfig(
                source=PatchingElementConfig(
                    activation=ModelLayerLocation.RESIDUAL_POST,
                    location=location,
                    filter=index_filter,
                    type=source_type,
                    value=value,
                ),
                target=PatchingElementConfig(
                    activation=ModelLayerLocation.RESIDUAL_POST,
                    location=location,
                    filter=index_filter,
                    type=PatchingType.REPLACE,
                ),
            )
        ],
    )


def _self_scale(model: Model, method: str, scale: float, layer: int = _GRADIENT_LAYER):
    """Scale one layer's residual by ``scale`` and write it back over itself.

    Both prompts are the target one, so the source activation the scale multiplies is the target's
    own: the patch writes ``s * a`` over ``a`` and the perturbation is exactly ``(s - 1) * a``,
    which is what makes it shrink to nothing as ``s`` approaches 1.
    """
    return _run(
        model,
        _gradient_config(method, source_type=PatchingType.VALUE_SCALE, value=scale, layer=layer),
        source=TARGET_PROMPT,
        correct_token=" Madrid",
        distractor_token=" Barcelona",
    ).squeeze()


def test_attribution_patching_is_a_directional_derivative(model: Model) -> None:
    torch.manual_seed(42)

    # The gradient is read once, at the target's own activation, so ten times the perturbation is
    # ten times the score exactly. The exact method reruns the model and is under no such obligation.
    small, large = (
        _self_scale(model, "attribution", 1.1),
        _self_scale(model, "attribution", 2.0),
    )
    assert torch.allclose(large, 10 * small, rtol=1e-3)

    exact_small, exact_large = (
        _self_scale(model, "exact", 1.1),
        _self_scale(model, "exact", 2.0),
    )
    assert not torch.allclose(exact_large, 10 * exact_small, rtol=1e-2)


def test_attribution_patching_reads_the_models_own_gradient(model: Model) -> None:
    torch.manual_seed(42)

    # The estimate is the derivative along the residual, so the model's own central difference is
    # what it has to reproduce: scaling up and scaling down by the same step, halved. Taking both
    # sides cancels the curvature that makes a one-sided difference disagree, leaving a comparison
    # the first-order estimate has to pass to within the step's cube.
    step = 0.05
    up = _self_scale(model, "exact", 1 + step)
    down = _self_scale(model, "exact", 1 - step)
    estimated = _self_scale(model, "attribution", 1 + step)

    assert torch.allclose(estimated, (up - down) / 2, rtol=0.05)


def test_scaling_the_final_residual_moves_neither_the_run_nor_its_estimate(model: Model) -> None:
    torch.manual_seed(42)

    # ``ln_final`` normalizes a uniform scaling of the last layer's residual away, so doubling it is
    # exactly a no-op -- and the derivative along the residual itself vanishes for the same reason.
    # The two methods have to agree here for reasons that have nothing to do with either's numerics.
    for method in ("exact", "attribution"):
        assert torch.allclose(_self_scale(model, method, 2.0, layer=-1), torch.tensor(0.0), atol=1e-4), method


def test_a_patch_writing_back_what_was_there_scores_zero(model: Model) -> None:
    torch.manual_seed(42)

    # The perturbation is identically zero, so every method reports nothing -- the exact one because
    # the run is unchanged, the gradient ones because they dot their gradient with it.
    for method in ("exact", "attribution", "integrated_gradients"):
        result = _run(
            model,
            _gradient_config(method),
            source=TARGET_PROMPT,
            correct_token=" Madrid",
            distractor_token=" Barcelona",
        )
        assert torch.allclose(result.squeeze(), torch.tensor(0.0), atol=1e-6), method


def test_integrated_gradients_of_one_step_is_attribution_patching(model: Model) -> None:
    torch.manual_seed(42)

    # A single step puts the only interpolation point at the clean activation, which is exactly
    # where attribution patching reads its gradient.
    estimated = _run(
        model, _gradient_config(IntegratedGradientsConfig(type="integrated_gradients", interpolation_steps=1))
    )

    assert torch.equal(estimated, _run(model, _gradient_config("attribution")))


def test_integrated_gradients_beats_a_single_gradient_on_a_full_resample(model: Model) -> None:
    torch.manual_seed(42)

    # Resampling a whole layer's residual is far outside the range a gradient at the clean point
    # describes; averaging along the path back to it is what recovers the effect.
    exact = _run(model, _gradient_config("exact"))
    attribution = _run(model, _gradient_config("attribution"))
    integrated = _run(model, _gradient_config("integrated_gradients"))

    assert abs(integrated - exact) < abs(attribution - exact)


def test_integrated_gradients_scores_a_row_the_same_whatever_else_the_index_sweeps(model: Model) -> None:
    torch.manual_seed(42)

    # Interpolating every hook on one pass would have each later hook scale the gradient reaching an
    # earlier one by the step's alpha, and move it off its own path, so a row's score would depend on
    # which other layers the sweep holds.
    layers = [0, model.cfg.n_layers // 2, model.cfg.n_layers - 1]
    method = IntegratedGradientsConfig(type="integrated_gradients", interpolation_steps=3)

    swept = _run(model, _gradient_config(method, layer=layers))
    alone = torch.cat([_run(model, _gradient_config(method, layer=layer)) for layer in layers])

    assert torch.allclose(swept, alone, rtol=1e-4, atol=1e-6)


def test_a_gradient_method_runs_the_model_once_however_large_the_index(model: Model, monkeypatch) -> None:
    torch.manual_seed(42)

    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        type=PatchingType.REPLACE,
    )
    calls: list[int] = []
    run_with_hooks = model.run_with_hooks
    monkeypatch.setattr(
        model, "run_with_hooks", lambda *args, **kwargs: (calls.append(1), run_with_hooks(*args, **kwargs))[1]
    )

    passes = {}
    for method in (
        EXACT,
        PatchingMethodConfig(type="attribution"),
        IntegratedGradientsConfig(type="integrated_gradients", interpolation_steps=4),
    ):
        config = replace(
            _patch_config(element, config_id=method.type),
            method=method,
            metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]},
        )
        calls.clear()
        collector = _collect(model, config)
        passes[method.type] = len(calls)
        assert len(collector.result) == model.cfg.n_layers

    # One patched run per row for the exact method; one pass whatever the index holds for a single
    # gradient. The integrated one interpolates each hook on passes of its own, one per step but the
    # clean point, where every hook is left as it is and one pass serves them all.
    assert passes == {"exact": model.cfg.n_layers, "attribution": 1, "integrated_gradients": 3 * model.cfg.n_layers + 1}


def _assert_rejected_before_any_run(model: Model, monkeypatch, patching_input: PatchingInput, message: str) -> None:
    """The plan refuses the input, and so does the run, before it makes a single forward pass."""
    runs = record_runs(monkeypatch, model)

    with pytest.raises(ValueError, match=message):
        Patcher(model).plan(patching_input, verbose=False)
    with pytest.raises(ValueError, match=message):
        Patcher(model).apply(patching_input, verbose=False)

    assert runs == []


def test_integrated_gradients_needs_a_source_run_to_interpolate_towards(model: Model, monkeypatch) -> None:
    config = replace(
        _last_layer_resid_zero_ablation_config(),
        method=IntegratedGradientsConfig(type="integrated_gradients"),
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]},
    )
    patching_input = PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona")

    _assert_rejected_before_any_run(model, monkeypatch, patching_input, "needs a source prompt")


def test_integrated_gradients_needs_the_two_prompts_to_line_up(model: Model, monkeypatch) -> None:
    # The interpolation runs position by position, so a shorter source prompt has nowhere to land.
    patching_input = PatchingInput(_gradient_config("integrated_gradients"), "Paris", TARGET_PROMPT)

    _assert_rejected_before_any_run(model, monkeypatch, patching_input, "same length")


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("guesswork", "Unknown patching method: guesswork"),
        # The interpolation steps are the integrated gradients' field; attribution takes none.
        (
            {"type": "attribution", "interpolation_steps": 4},
            r"Unknown keys for PatchingMethodConfig: \['interpolation_steps'\]",
        ),
        ({"type": "integrated_gradients", "interpolation_steps": 0}, "interpolation_steps must be at least 1, got 0"),
        ({"interpolation_steps": 4}, r"Missing required keys for PatchingMethodConfig: \['type'\]"),
    ],
)
def test_a_method_or_method_config_that_does_not_fit_is_rejected_as_the_config_is_read(
    config_from_yaml, method: str | dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_yaml(PatchingConfig, "unfit", _patch_yaml(method=method))


# --------------------------------------------------------------------------------------------------
# Where a patch writes - every kind of activation, patched by a run and by hand
# --------------------------------------------------------------------------------------------------
_EVERY_SLICE = slice(None)


@pytest.mark.parametrize(
    ("coordinate", "writes"),
    [
        # [batch, position, neuron]
        ({"activation": "embed", "position": 4}, [("hook_embed", (4,))]),
        ({"activation": "resid.mid", "layer": 5, "position": 4}, [("blocks.5.hook_resid_mid", (4,))]),
        ({"activation": "resid.post", "layer": 3, "position": [3, 4]}, [("blocks.3.hook_resid_post", ([3, 4],))]),
        ({"activation": "attn.out", "layer": 9, "position": -1}, [("blocks.9.hook_attn_out", (-1,))]),
        (
            {"activation": "mlp.post", "layer": 9, "position": -1, "neuron": 373},
            [("blocks.9.mlp.hook_post", (-1, 373))],
        ),
        (
            {"activation": "mlp.post", "layer": 9, "position": -1, "neuron": [373, 374]},
            [("blocks.9.mlp.hook_post", (-1, [373, 374]))],
        ),
        # [batch, position, head, neuron]
        ({"activation": "attn.z", "layer": 9, "position": -1, "head": 8}, [("blocks.9.attn.hook_z", (-1, 8))]),
        (
            {"activation": "attn.z", "layer": 9, "position": -1, "head": [8, 9]},
            [("blocks.9.attn.hook_z", (-1, [8, 9]))],
        ),
        (
            {"activation": "attn.z", "layer": 9, "position": -1, "head": 8, "neuron": 11},
            [("blocks.9.attn.hook_z", (-1, 8, 11))],
        ),
        ({"activation": "attn.q", "layer": 9, "position": -1, "head": 8}, [("blocks.9.attn.hook_q", (-1, 8))]),
        ({"activation": "attn.k", "layer": 9, "position": 4, "head": 8}, [("blocks.9.attn.hook_k", (4, 8))]),
        ({"activation": "attn.v", "layer": 9, "position": 4, "head": 8}, [("blocks.9.attn.hook_v", (4, 8))]),
        # [batch, head, dest_position, src_position]
        ({"activation": "attn.pattern", "layer": 9, "head": 8}, [("blocks.9.attn.hook_pattern", (8,))]),
    ],
    ids=lambda value: value["activation"] if isinstance(value, dict) else "",
)
def test_a_patch_writes_exactly_where_its_coordinate_says(model: Model, coordinate: dict, writes: list) -> None:
    result = _run(model, _patch_config(_explicit_element([coordinate], activation=None))).squeeze()

    expected = _expected_logit_diff(model, _copying_hooks(model, writes))
    assert torch.allclose(result, expected, rtol=0, atol=1e-6)
    # The patch moves the metric, so one written anywhere else would show.
    assert abs(float(expected - _unpatched_metrics(model, TARGET_PROMPT)[PatchingMetric.LOGIT_DIFF])) > 1e-4


@pytest.mark.parametrize(
    ("model_fixture", "coordinate", "writes"),
    [
        # Grouped-query attention keys and values have a head axis of their own, the key/value heads.
        ("gqa_model", {"activation": "attn.k", "layer": 1, "head": 1}, [("blocks.1.attn.hook_k", (_EVERY_SLICE, 1))]),
        ("gqa_model", {"activation": "attn.v", "layer": 0, "head": 1}, [("blocks.0.attn.hook_v", (_EVERY_SLICE, 1))]),
        # A gated MLP's linear branch is a location of its own.
        (
            "gated_model",
            {"activation": "mlp.pre_linear", "layer": 0, "position": 4},
            [("blocks.0.mlp.hook_pre_linear", (4,))],
        ),
    ],
    ids=lambda value: value["activation"] if isinstance(value, dict) else "",
)
def test_a_patch_writes_where_its_coordinate_says_on_other_architectures(
    request, backend: str, model_fixture: str, coordinate: dict, writes: list
) -> None:
    # Random weights predict the same token for both prompts, so the answers are named.
    model = request.getfixturevalue(model_fixture)
    answers = {"correct_token": " Paris", "distractor_token": " Madrid"}
    ids = (model.to_single_token(" Paris"), model.to_single_token(" Madrid"))

    result = _run(model, _patch_config(_explicit_element([coordinate], activation=None)), **answers).squeeze()

    expected = _expected_logit_diff(model, _copying_hooks(model, writes), answers=ids)
    assert torch.allclose(result, expected, rtol=0, atol=1e-7)
    assert abs(float(expected - _expected_logit_diff(model, [], answers=ids))) > 1e-5


@pytest.mark.parametrize(
    "element",
    [
        # Two heads grouped into one row, beside a third left on its own.
        {"location": ["layer", "head"], "filter": [{"layer": 9, "head": [7, 8, 9]}], "group": {"head": [8, 9]}},
        # A growing set of heads, one more per row.
        {"location": ["layer", "head"], "filter": [{"layer": 9, "head": 7}], "incremental": {"head": [8, 9]}},
        # Every pair of three heads, and each on its own.
        {
            "location": ["layer", "head"],
            "filter": [{"layer": 9, "head": [7, 8, 9]}],
            "group": {"head": "all"},
            "combination": {"location": ["head"], "max_elements": 2},
        },
        {"coordinates": [{"layer": 9, "head": [7, 8]}, {"layer": 9, "head": 9}]},
    ],
    ids=["group", "incremental", "combination", "coordinates"],
)
def test_every_row_patches_the_heads_it_names_together(model: Model, config_from_yaml, element: dict) -> None:
    element = {"activation": "attn.z", **element, "type": "replace"}
    config = config_from_yaml(
        PatchingConfig,
        "head_sets",
        {"metric": {"logit_diff": ["absolute"]}, "method": "exact", "patch": [{"source": element, "target": element}]},
    )

    result = _collect(model, config).result

    assert len(result) > 1
    for _, row in result.iterrows():
        heads = row["source_head"] if isinstance(row["source_head"], list | tuple) else [row["source_head"]]
        expected = _expected_logit_diff(
            model, _copying_hooks(model, [("blocks.9.attn.hook_z", (_EVERY_SLICE, list(heads)))])
        )
        assert row["logit_diff"] == pytest.approx(float(expected), abs=1e-6), heads
    # Every set of heads scores apart from every other, so a row that patched another set would show.
    assert result["logit_diff"].nunique() == len(result)


# --------------------------------------------------------------------------------------------------
# Explicit coordinates
# --------------------------------------------------------------------------------------------------
def _explicit_element(
    coordinates: list[dict], activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST
) -> PatchingElementConfig:
    return PatchingElementConfig(
        activation=activation,
        coordinates=coordinates,
        type=PatchingType.REPLACE,
    )


def _explicit_patch_config(
    source_coordinates: list[dict],
    target_coordinates: list[dict],
    activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST,
) -> PatchingConfig:
    """Patch the exact coordinates listed: source coordinate i is read from the source run and written at
    target coordinate i of the target run (one patched run per pair)."""
    return PatchingConfig(
        config_id="explicit_patch",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED]},
        method=EXACT,
        patch=[
            SinglePatchingConfig(
                source=_explicit_element(source_coordinates, activation),
                target=_explicit_element(target_coordinates, activation),
            )
        ],
    )


def _run_explicit(
    model: Model,
    source_coordinates: list[dict],
    target_coordinates: list[dict],
    save_dir: str | None = None,
    activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST,
) -> torch.Tensor:
    patching_input = PatchingInput(
        _explicit_patch_config(source_coordinates, target_coordinates, activation), SOURCE_PROMPT, TARGET_PROMPT
    )
    collector = Patcher(model).apply(patching_input, save_dir=save_dir, return_collector=True, verbose=False)
    assert collector is not None
    return _metric(collector)


def test_explicit_coordinates_matches_declarative_patch(model: Model) -> None:
    torch.manual_seed(42)

    # The explicit twin of ``test_activation_patching_from_source_into_target``: one coordinate covering
    # the last layer's full residual stream reproduces the declarative filter's result.
    coordinate = [{ModelAxis.LAYER: -1}]
    result = _run_explicit(model, coordinate, coordinate)

    assert result.shape == (1,)
    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_explicit_position_is_respected(model: Model) -> None:
    torch.manual_seed(42)

    # The final token's logits depend only on the last position of the final layer's residual, so
    # patching just that coordinate reproduces the full source run (metric 1.0). The prompts share their
    # prefix, so position 0 carries identical activations in both runs and patching it is a no-op
    # (metric 0.0). If the position were ignored (patching every position), both would give 1.0.
    last_layer_last_position = [{ModelAxis.LAYER: -1, ModelAxis.POSITION: -1}]
    last_layer_first_position = [{ModelAxis.LAYER: -1, ModelAxis.POSITION: 0}]
    last_position = _run_explicit(model, last_layer_last_position, last_layer_last_position)
    first_position = _run_explicit(model, last_layer_first_position, last_layer_first_position)

    assert torch.allclose(last_position.squeeze(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(first_position.squeeze(), torch.tensor(0.0), atol=1e-5)


def test_explicit_pairing_and_artifacts(model: Model, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Two pairs reading different source layers into different target positions of layer 7.
    source_coordinates = [
        {ModelAxis.LAYER: 5, ModelAxis.POSITION: -1},
        {ModelAxis.LAYER: 6, ModelAxis.POSITION: -1},
    ]
    target_coordinates = [
        {ModelAxis.LAYER: 7, ModelAxis.POSITION: -1},
        {ModelAxis.LAYER: 7, ModelAxis.POSITION: -2},
    ]
    save_dir = tmp_path / "explicit_patch_run"
    paired = _run_explicit(model, source_coordinates, target_coordinates, save_dir=str(save_dir))

    # Explicit coordinates need not form a grid, so the result is flat: one entry per pair, in order,
    # each matching the same pair patched on its own.
    assert paired.shape == (2,)
    first = _run_explicit(model, source_coordinates[:1], target_coordinates[:1])
    second = _run_explicit(model, source_coordinates[1:], target_coordinates[1:])
    assert torch.allclose(paired, torch.cat([first, second]))
    assert not torch.allclose(first, second)

    # index-0.csv pairs each source coordinate with the target coordinate it was written into.
    index_df = pd.read_csv(save_dir / "index-0.csv")
    assert len(index_df) == 2
    assert index_df["source_layer"].to_list() == [5, 6]
    assert index_df["target_layer"].to_list() == [7, 7]
    assert index_df["target_position"].nunique() == 2
    # The locations the pairs name and nothing else - how they are laid out on a plot is not the
    # index file's business.
    assert all(col.startswith(("source_", "target_")) for col in index_df.columns)


def test_explicit_per_coordinate_activation(model: Model) -> None:
    torch.manual_seed(42)

    # Each coordinate carries its own activation: pair 0 patches the final layer's residual at the last
    # position, reproducing the full source run (metric exactly 1.0, as established above); pair 1
    # patches mlp.out at the same coordinate, which only swaps the MLP component's contribution, so its
    # metric lands well away from 1.0 -- proving each row is patched at its own activation's hook.
    coordinates = [
        {
            ModelAxis.ACTIVATION: ModelLayerLocation.RESIDUAL_POST,
            ModelAxis.LAYER: -1,
            ModelAxis.POSITION: -1,
        },
        {ModelAxis.ACTIVATION: ModelLayerLocation.MLP_OUT, ModelAxis.LAYER: -1, ModelAxis.POSITION: -1},
    ]
    result = _run_explicit(model, coordinates, coordinates, activation=None)

    assert result.shape == (2,)
    assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(result[1], torch.tensor(1.0), atol=1e-2)


def test_two_sources_differing_only_in_activation_keep_their_own_metric(model: Model) -> None:
    torch.manual_seed(42)

    # Explicit source coordinates against a declarative target: that mix is what sends the index
    # through the factorized output branch, which drops the activation column. Both coordinates sit at
    # the same layer/position/neuron, so activation is the only thing telling the two runs apart --
    # and a collector keyed on that coordinate would file both under one cell, reporting whichever ran
    # last on both rows.
    source = PatchingElementConfig(
        type=PatchingType.REPLACE,
        coordinates=[
            {
                ModelAxis.ACTIVATION: activation,
                ModelAxis.LAYER: 0,
                ModelAxis.POSITION: -1,
                ModelAxis.NEURON: 1,
            }
            for activation in (ModelLayerLocation.MLP_POST, ModelLayerLocation.MLP_PRE)
        ],
    )
    # One target row, so it is held steady while the two source coordinates vary.
    target = PatchingElementConfig(
        type=PatchingType.REPLACE,
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: 0}]),
    )
    config = PatchingConfig(
        config_id="same_coordinate_two_activations",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )
    patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)
    assert collector is not None
    result = collector.result

    # The two coordinates are one grid point once the activation is set aside...
    locations = [f"source_{location.value}" for location in (ModelAxis.LAYER, ModelAxis.POSITION)]
    assert len(result[[*locations, "source_neuron"]].drop_duplicates()) == 1
    # ...yet each run keeps its own row, naming the activation it read and the metric it produced.
    assert result[f"source_{ModelAxis.ACTIVATION.value}"].to_list() == [
        ModelLayerLocation.MLP_POST,
        ModelLayerLocation.MLP_PRE,
    ]
    assert _metric(collector)[0] != _metric(collector)[1]


def test_artifacts_saved_to_disk(model: Model, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Build the config through ConfigFactory by persisting it to a YAML file and loading it back,
    # mirroring how configs are sourced in production rather than constructing PatchingConfig directly.
    element = {
        "activation": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": -1}],
        "type": "replace",
    }
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        yaml.safe_dump(
            {
                "last_layer_resid_patch": {
                    "metric": {"logit_diff": ["absolute"]},
                    "method": "exact",
                    "patch": [{"source": element, "target": element}],
                }
            }
        )
    )
    config = ConfigFactory[PatchingConfig](str(config_yaml)).create("last_layer_resid_patch")
    patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)

    save_dir = tmp_path / "patch_run"
    collector = Patcher(model).apply(patching_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    # The three artifacts the collector is expected to persist.
    result_path = save_dir / "result.csv"
    index_path = save_dir / "index-0.csv"
    config_path = save_dir / "config.json"
    assert result_path.exists()
    assert index_path.exists()
    assert config_path.exists()
    assert not (save_dir / "result.pt").exists()

    # result.csv round-trips the in-memory frame: the source/target columns of the patched element,
    # plus one column per metric, named after it...
    result_df = pd.read_csv(result_path, keep_default_na=False)
    assert len(result_df) == len(collector.result) == 1
    assert result_df["logit_diff"].to_list() == pytest.approx(collector.result["logit_diff"].to_list())
    # ...then the token the patched run predicts next, as its id and its string.
    assert result_df["generation"].to_list() == collector.result["generation"].to_list()
    assert result_df["str_generation"].to_list() == collector.result["str_generation"].to_list()
    assert any(col.startswith("source_") for col in result_df.columns)
    assert any(col.startswith("target_") for col in result_df.columns)

    # config.json mirrors the patching config that produced the run.
    saved_config = json.loads(config_path.read_text())
    assert saved_config["config_id"] == "last_layer_resid_patch"
    assert saved_config["metric"] == {"logit_diff": ["absolute"]}
    assert saved_config["method"] == {"type": "exact"}
    assert len(saved_config["patch"]) == 1
    assert saved_config["patch"][0]["source"]["type"] == "replace"
    assert saved_config["patch"][0]["target"]["type"] == "replace"

    # index-0.csv carries one row per result entry, with the source/target columns.
    index_df = pd.read_csv(index_path)
    assert len(index_df) == len(collector.result)
    assert any(col.startswith("source_") for col in index_df.columns)
    assert any(col.startswith("target_") for col in index_df.columns)


def _gaussian_noise_config() -> PatchingConfig:
    """Add seeded Gaussian noise to the target's own residual, then write it back over itself."""
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: _VALUE_SCALE_LAYER}]),
        type=PatchingType.GAUSSIAN_NOISE,
    )
    target = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: _VALUE_SCALE_LAYER}]),
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="gaussian_noise",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def test_gaussian_noise_patching(model: Model) -> None:
    torch.manual_seed(42)

    noised = _run(model, _gaussian_noise_config())

    # The strategy draws from a seeded generator, so the run repeats exactly...
    assert noised.shape == (1,)
    assert torch.isfinite(noised).all()
    assert torch.equal(_run(model, _gaussian_noise_config()), noised)
    # ...and noising the activation is not the same as replacing it with the source's.
    assert not torch.allclose(noised, _run(model, _resid_value_scale_config(1.0, _VALUE_SCALE_LAYER)))


def test_gaussian_noise_is_added_at_three_standard_deviations_of_the_activation(model: Model, monkeypatch) -> None:
    # The noised residual is read where the patched run carries it on, as the next block's input, so
    # what is checked is what the model ran on.
    layer = _VALUE_SCALE_LAYER
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(TARGET_PROMPT))
    clean = cache[f"blocks.{layer}.hook_resid_post"][0]
    downstream: list[torch.Tensor] = []
    run_with_hooks = model.run_with_hooks

    def recording(*args, fwd_hooks=(), **kwargs):
        record = (f"blocks.{layer + 1}.hook_resid_pre", lambda activation, hook: downstream.append(activation.clone()))
        return run_with_hooks(*args, fwd_hooks=[*fwd_hooks, record], **kwargs)

    monkeypatch.setattr(model, "run_with_hooks", recording)
    _run(model, _gaussian_noise_config())

    (noised,) = downstream
    noise = noised[0] - clean
    scale = 3 * float(clean.std())

    # Zero-mean, at the scale the corruption is defined by, and drawn independently of the
    # activation it lands on - not a rescaled copy of it.
    assert abs(float(noise.mean())) < 0.05 * scale
    assert float(noise.std()) == pytest.approx(scale, rel=0.05)
    assert abs(float(torch.corrcoef(torch.stack([noise.flatten(), clean.flatten()]))[0, 1])) < 0.1


@pytest.mark.parametrize("source_type", list(PatchingType))
def test_a_batched_run_scores_what_an_unbatched_one_does(model: Model, source_type: PatchingType) -> None:
    # A patched run holds one copy of the target prompt per row; every row still reads its source from
    # the one prompt each cache keeps. Five rows a batch leaves a ragged last one, padded to five.
    def element(patching_type: PatchingType) -> PatchingElementConfig:
        return PatchingElementConfig(
            activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelAxis.LAYER], type=patching_type, value=2.0
        )

    config = PatchingConfig(
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element(source_type), target=element(PatchingType.REPLACE))],
    )

    unbatched = _batched(model, config, batch_size=1)
    batched = _batched(model, config, batch_size=5)

    assert len(unbatched.result) == model.cfg.n_layers
    assert torch.allclose(_metric(batched), _metric(unbatched), atol=1e-4)
    assert batched.result["generation"].tolist() == unbatched.result["generation"].tolist()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_a_one_row_element_is_patched_into_every_row_of_a_batch(model: Model, batch_size: int) -> None:
    # The sweep patches the BOS position, which both prompts share, so on its own it changes nothing;
    # the one-row companion swaps the country token in on every run. Every row is then the
    # companion's effect alone - in each row of a batch, not only the one whose hooks were built last.
    sweep = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=Filter(inclusion_terms=[{ModelAxis.POSITION: 0}]),
        type=PatchingType.REPLACE,
    )
    companion = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_PRE,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: 0, ModelAxis.POSITION: 4}]),
        type=PatchingType.REPLACE,
    )
    metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]}
    alone = _collect(
        model,
        PatchingConfig(metric=metric, method=EXACT, patch=[SinglePatchingConfig(source=companion, target=companion)]),
    )
    together = _batched(
        model,
        PatchingConfig(
            metric=metric,
            method=EXACT,
            patch=[
                SinglePatchingConfig(source=sweep, target=sweep),
                SinglePatchingConfig(source=companion, target=companion),
            ],
        ),
        batch_size=batch_size,
    )

    # The companion moves the prediction, so a row it missed would show.
    assert alone.result["generation"].tolist() != [_generations(model)[1]]
    assert len(together.result) == model.cfg.n_layers
    assert torch.allclose(_metric(together), _metric(alone).expand(model.cfg.n_layers), atol=1e-4)
    assert together.result["generation"].tolist() == alone.result["generation"].tolist() * model.cfg.n_layers


@pytest.fixture
def batch_shape_dependent_model(model: Model):
    """``model`` with numerics that depend on the batch shape, as MPS kernels' do in half precision.

    CPU kernels round the same at every batch size, so without this a run compared against a
    reference made at another shape is indistinguishable from one compared at its own. The shift is
    along one coordinate: LayerNorm centers its input, so the same shift on every coordinate would
    cancel up to rounding, and runs at different widths could still agree bit for bit.
    """

    def shift(activation, hook):
        activation = activation.clone()
        activation[..., 0] += 1e-3 * activation.shape[0]
        return activation

    point = model.hook_dict["hook_embed"]
    assert not point.fwd_hooks
    point.add_hook(shift, dir="fwd", is_permanent=True)
    yield model
    point.remove_hooks("fwd", including_permanent=True)


def _self_patch_config() -> PatchingConfig:
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        metric={PatchingMetric.PROBABILITY: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )


@pytest.mark.parametrize("batch_size", [1, 5])
def test_patching_a_prompt_into_itself_reproduces_the_target_run(
    batch_shape_dependent_model: Model, batch_size: int
) -> None:
    # Writing back what a run already holds changes nothing, so every row must read exactly what the
    # unpatched run does - which holds only if that run and its patch values were made at the patched
    # runs' own batch shape. Five rows a batch leaves a ragged last one.
    patching_input = PatchingInput(_self_patch_config(), TARGET_PROMPT, TARGET_PROMPT)
    collector = Patcher(batch_shape_dependent_model, collector_class=PatchingCollector, batch_size=batch_size).apply(
        patching_input, return_collector=True, verbose=False
    )
    assert collector is not None

    assert len(collector.result) % batch_size != 0 or batch_size == 1
    # The unpatched run, made as wide as the patched ones.
    with torch.no_grad():
        logits = batch_shape_dependent_model(
            torch.cat([batch_shape_dependent_model.to_tokens(TARGET_PROMPT)] * batch_size)
        )[0, -1]
    generation = int(logits.argmax())
    assert collector.result["probability"].tolist() == [logits.softmax(dim=-1)[generation].item()] * len(
        collector.result
    )
    assert collector.result["generation"].tolist() == [generation] * len(collector.result)


def test_an_input_run_at_another_batch_size_remakes_its_runs_at_that_width(
    batch_shape_dependent_model: Model, monkeypatch
) -> None:
    # The unpatched runs a patch is scored against are made as wide as the patched ones, so an input
    # run again under another batch size has to run its prompts again at the new width - once.
    model = batch_shape_dependent_model

    def probabilities(patching_input: PatchingInput, batch_size: int) -> list[float]:
        patcher = Patcher(model, collector_class=PatchingCollector, batch_size=batch_size)
        return patcher.apply(patching_input, return_collector=True, verbose=False).result["probability"].tolist()

    fresh = probabilities(PatchingInput(_self_patch_config(), SOURCE_PROMPT, TARGET_PROMPT), batch_size=3)
    reused = PatchingInput(_self_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)
    narrow = probabilities(reused, batch_size=1)

    runs = record_runs(monkeypatch, model)
    wide = probabilities(reused, batch_size=3)
    again = probabilities(reused, batch_size=3)

    assert wide == again == fresh
    assert wide != narrow
    # Both prompts once at the new width, and not again at the same one.
    assert [run.width for run in runs if run.entry == "run_with_cache"] == [3, 3]


def test_a_batch_wider_than_the_sweep_runs_only_as_wide_as_its_rows(model: Model, monkeypatch) -> None:
    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)
    runs = record_runs(monkeypatch, model)

    Patcher(model, collector_class=PatchingCollector, batch_size=16).apply(patching_input, verbose=False)

    # One row to patch: the two unpatched runs and the patched one are each a single prompt wide.
    assert [(run.entry, run.width) for run in runs] == [("run_with_cache", 1)] * 2 + [("run_with_hooks", 1)]


def _batched(model: Model, config: PatchingConfig, batch_size: int) -> PatchingCollector:
    collector = Patcher(model, collector_class=PatchingCollector, batch_size=batch_size).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector


def _layer_sweep(layers: list[int], position: int = -1) -> PatchingElementConfig:
    return PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: layers, ModelAxis.POSITION: position}]),
        type=PatchingType.REPLACE,
    )


def _pair_config(*pairs: tuple[PatchingElementConfig, PatchingElementConfig]) -> PatchingConfig:
    return PatchingConfig(
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target) for source, target in pairs],
    )


def test_the_result_is_laid_out_along_the_element_with_the_most_rows(model: Model) -> None:
    sweep = _layer_sweep(list(range(model.cfg.n_layers)))
    companion = _layer_sweep([0], position=1)

    companion_first = _collect(model, _pair_config((companion, companion), (sweep, sweep)))
    sweep_first = _collect(model, _pair_config((sweep, sweep), (companion, companion)))

    assert len(companion_first.result) == model.cfg.n_layers
    assert companion_first.result["target_layer"].tolist() == list(range(model.cfg.n_layers))
    assert torch.equal(_metric(companion_first), _metric(sweep_first))


def test_a_source_and_a_target_pair_row_by_row_in_the_result(model: Model) -> None:
    # Both filters leave gaps in the frame's row labels; the rows still pair in order.
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=Filter(
            inclusion_terms=[
                {ModelAxis.LAYER: 0, ModelAxis.POSITION: 1},
                {ModelAxis.LAYER: [1, 2], ModelAxis.POSITION: 2},
            ]
        ),
        type=PatchingType.REPLACE,
    )
    target = _layer_sweep([0, 1, 2], position=3)

    result = _collect(model, _pair_config((source, target))).result

    assert not result.isna().any().any()
    assert result["source_layer"].tolist() == [0, 1, 2]
    assert result["source_position"].tolist() == [1, 2, 2]
    assert result["target_layer"].tolist() == [0, 1, 2]
    assert result["target_position"].tolist() == [3, 3, 3]


def test_a_one_row_source_is_written_into_every_row_of_the_target(model: Model, tmp_path: Path) -> None:
    source, target = _layer_sweep([0], position=1), _layer_sweep([0, 1, 2], position=1)
    save_dir = tmp_path / "broadcast"

    Patcher(model).apply(
        PatchingInput(_pair_config((source, target)), SOURCE_PROMPT, TARGET_PROMPT),
        save_dir=str(save_dir),
        verbose=False,
    )

    for frame in (pd.read_csv(save_dir / "result.csv"), pd.read_csv(save_dir / "index-0.csv")):
        assert frame["source_layer"].tolist() == [0, 0, 0]
        assert frame["target_layer"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([((0, 1, 2), (0, 1, 2, 3, 4))], "source of 3 rows cannot be paired with a target of 5"),
        ([((0, 1, 2), (0, 1, 2)), ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))], "3 and 5 rows"),
    ],
)
def test_indices_that_cannot_pair_row_by_row_are_rejected_before_a_run(
    model: Model, monkeypatch, pairs: list, message: str
) -> None:
    config = _pair_config(*((_layer_sweep(list(source)), _layer_sweep(list(target))) for source, target in pairs))

    forbid_model_runs(monkeypatch, model)
    with pytest.raises(ValueError, match=message):
        Patcher(model).plan(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


@pytest.mark.parametrize("source_type", [PatchingType.REPLACE, PatchingType.VALUE_SCALE])
def test_a_source_that_reads_the_source_run_needs_a_source_prompt(source_type: PatchingType) -> None:
    source = replace(_last_layer_resid_element(), type=source_type)
    config = _pair_config((source, _last_layer_resid_element()))

    with pytest.raises(ValueError, match=f"{source_type.value} reads the source run, which needs a source prompt"):
        PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona")


@pytest.mark.parametrize("method", ["exact", "attribution"])
def test_a_target_that_is_not_replaced_is_rejected_before_a_run(method: str) -> None:
    target = replace(_last_layer_resid_element(), type=PatchingType.VALUE_REPLACE)
    config = replace(_pair_config((_last_layer_resid_element(), target)), method=PatchingMethodConfig(type=method))

    with pytest.raises(ValueError, match="Unknown target strategy: value_replace"):
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)


def test_a_run_without_patch_elements_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one patch element"):
        PatchingInput(_pair_config(), SOURCE_PROMPT, TARGET_PROMPT)


def test_exact_patching_builds_no_autograd_graph(model: Model, monkeypatch) -> None:
    tracked: list[bool] = []
    run_with_hooks = model.run_with_hooks

    def recording(*args, **kwargs):
        logits = run_with_hooks(*args, **kwargs)
        tracked.append(logits.requires_grad)
        return logits

    monkeypatch.setattr(model, "run_with_hooks", recording)
    _collect(model, _last_layer_resid_patch_config())

    assert tracked == [False]


@pytest.mark.parametrize("method", ["attribution", "integrated_gradients"])
def test_a_gradient_method_runs_with_autograd_disabled(model: Model, method: str) -> None:
    config = _gradient_config(method)
    enabled = _run(model, config)

    with torch.no_grad():
        disabled = _run(model, config)

    assert torch.allclose(disabled, enabled)


@pytest.mark.parametrize("method", ["attribution", "integrated_gradients"])
def test_a_gradient_method_runs_on_a_frozen_model(backend: str, method: str) -> None:
    model = build_small_model(backend)
    config = _gradient_config(method, layer=1)
    answers = {"correct_token": " Paris", "distractor_token": " Madrid"}
    trainable = _run(model, config, **answers)

    model.requires_grad_(False)
    frozen = _run(model, config, **answers)

    assert torch.allclose(frozen, trainable)


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ("source", "Unknown source strategy"),
        ("target", "Unknown target strategy"),
    ],
)
def test_a_patching_type_that_is_not_known_raises(model: Model, element: str, message: str) -> None:
    # A config built in code can name a type no strategy implements - and 'value_replace' is one only
    # a source may carry, so a target is rejected the same way.
    source, target = _last_layer_resid_element(), _last_layer_resid_element()
    (source if element == "source" else target).type = "bogus" if element == "source" else "value_replace"
    config = PatchingConfig(
        config_id="unknown_strategy",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )

    with pytest.raises(ValueError, match=message):
        Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


def test_a_patching_metric_that_is_not_known_raises(model: Model) -> None:
    config = _last_layer_resid_patch_config()
    config.metric = {"accuracy": [MetricComparison.ABSOLUTE]}

    with pytest.raises(ValueError, match="Unknown patching metric"):
        Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


def test_a_metric_comparison_that_is_not_known_raises(model: Model) -> None:
    config = _last_layer_resid_patch_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: ["ratio"]}

    with pytest.raises(ValueError, match="Unknown metric comparison"):
        Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"metric": {}}, "At least one patching metric"),
        ({"metric": {PatchingMetric.LOGIT_DIFF: []}}, "logit_diff names no comparison"),
        (
            {"metric": {PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE, MetricComparison.ABSOLUTE]}},
            "logit_diff names absolute more than once",
        ),
    ],
)
def test_a_config_that_asks_for_nothing_measurable_is_rejected(overrides: dict, message: str) -> None:
    settings = {
        "config_id": "broken",
        "metric": {PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        "method": EXACT,
        "patch": [],
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        PatchingConfig(**settings)


def test_a_run_without_a_collector_still_writes_its_artifacts(model: Model, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Not asking for the collector is the default, and how the example scripts run an operation.
    collector = Patcher(model).apply(
        PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT),
        return_collector=True,
        verbose=False,
    )
    assert collector is not None

    save_dir = tmp_path / "patch_run"
    returned = Patcher(model).apply(
        PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT),
        save_dir=str(save_dir),
        verbose=False,
    )

    # Nothing comes back, but the run did all of its work and left it on disk.
    assert returned is None
    # keep_default_na=False: a generated token such as "null" is a string, not a missing value.
    pd.testing.assert_frame_equal(pd.read_csv(save_dir / "result.csv", keep_default_na=False), collector.result)
    assert (save_dir / "index-0.csv").exists()


# --------------------------------------------------------------------------------------------------
# Configuration - patching configs come from YAML, where a source and a target often alias one anchor
# --------------------------------------------------------------------------------------------------
ELEMENT_YAML = {"activation": "resid.post", "location": ["layer"], "filter": [{"layer": -1}], "type": "replace"}


def _patch_yaml(**overrides) -> dict:
    return {
        "metric": {"logit_diff": ["absolute"]},
        "method": "exact",
        "patch": [{"source": ELEMENT_YAML, "target": ELEMENT_YAML}],
        **overrides,
    }


def test_a_yaml_config_runs_the_same_way_a_code_built_one_does(model: Model, config_from_yaml) -> None:
    from_file = config_from_yaml(PatchingConfig, "last_layer_resid_patch", _patch_yaml())
    element = _last_layer_resid_element()
    in_code = PatchingConfig(
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )

    assert from_file.metric == {PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]}
    assert from_file.method == EXACT
    assert from_file.patch[0].source.activation is ModelLayerLocation.RESIDUAL_POST
    assert from_file.patch[0].source.location == [ModelAxis.LAYER]
    assert from_file.patch[0].source.type is PatchingType.REPLACE

    def run(config: PatchingConfig) -> torch.Tensor:
        patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)
        collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)
        assert collector is not None
        return _metric(collector)

    assert torch.equal(run(from_file), run(in_code))


def test_a_yaml_config_can_give_each_metric_its_own_comparisons(config_from_yaml) -> None:
    config = config_from_yaml(
        PatchingConfig,
        "attributed",
        _patch_yaml(
            metric={"logit_diff": ["normalized", "absolute"], "kl_divergence": ["indirect_effect"]},
            method={"type": "integrated_gradients", "interpolation_steps": 4},
        ),
    )

    assert config.metric == {
        PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED, MetricComparison.ABSOLUTE],
        PatchingMetric.KL_DIVERGENCE: [MetricComparison.INDIRECT_EFFECT],
    }
    assert config.method == IntegratedGradientsConfig(type="integrated_gradients", interpolation_steps=4)
    assert sorted(config.metric_columns()) == sorted(
        ["normalized_logit_diff", "logit_diff", "indirect_effect_kl_divergence"]
    )


def test_the_columns_follow_the_order_the_yaml_names_them(tmp_path: Path) -> None:
    # Written out rather than dumped through the fixture, which sorts the mapping's keys: the order
    # of the columns -- and so which one the figure is drawn from -- is the file's own.
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        "ordered:\n"
        "  metric:\n"
        "    kl_divergence: [indirect_effect]\n"
        "    logit_diff: [normalized, absolute]\n"
        "  method: exact\n"
        "  patch: []\n"
    )

    config = ConfigFactory[PatchingConfig](str(config_yaml)).create("ordered")

    assert config.metric_columns() == ["indirect_effect_kl_divergence", "normalized_logit_diff", "logit_diff"]
    assert config.primary() == (PatchingMetric.KL_DIVERGENCE, MetricComparison.INDIRECT_EFFECT)


def test_a_source_and_a_target_may_alias_the_same_yaml_anchor(model: Model, tmp_path: Path) -> None:
    # A YAML anchor gives the deserializer the very same mapping twice; it must not consume it.
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        "anchored:\n"
        "  metric: {logit_diff: [absolute]}\n"
        "  method: exact\n"
        "  patch:\n"
        "    - source: &element\n"
        "        activation: resid.post\n"
        "        location: [layer]\n"
        "        filter: [{layer: -1}]\n"
        "        type: replace\n"
        "      target: *element\n"
    )
    config = ConfigFactory[PatchingConfig](str(config_yaml)).create("anchored")

    assert config.patch[0].source == config.patch[0].target
    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    assert len(collector.result) == 1


def test_the_omitted_keys_of_an_element_fall_back_to_their_defaults(model: Model, config_from_yaml) -> None:
    # A source that names nothing but its type produces zeros - the zero-ablation shape, from YAML.
    raw = {
        "metric": {"logit_diff": ["absolute"]},
        "method": "exact",
        "patch": [{"source": {"type": "value_replace"}, "target": ELEMENT_YAML}],
    }
    config = config_from_yaml(PatchingConfig, "zero_ablation", raw)
    source = config.patch[0].source

    assert source.activation is None
    assert source.location == []
    assert (source.filter, source.group, source.incremental) == (Filter(), Group(), Incremental())
    assert source.value == 0.0

    # Those defaults are the zero ablation built in code, so the two runs agree.
    assert torch.equal(_run(model, config), _run(model, _last_layer_resid_zero_ablation_config()))


def test_an_exclusion_term_is_deserialized_into_its_own_filter(model: Model, config_from_yaml) -> None:
    element = {**ELEMENT_YAML, "filter": [{"layer": {"range": [0, 2]}}, {"not": {"layer": 1}}]}
    config = config_from_yaml(
        PatchingConfig,
        "filtered",
        _patch_yaml(patch=[{"source": element, "target": element}]),
    )

    assert config.patch[0].source.filter == Filter(
        inclusion_terms=[{ModelAxis.LAYER: {"range": [0, 2]}}], exclusion_terms=[{ModelAxis.LAYER: 1}]
    )

    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    assert len(collector.result) == 2  # layers 0 and 2 survive the exclusion


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_patch_yaml(typ="replace"), "Unknown keys for PatchingConfig.*typ"),
        (
            _patch_yaml(patch=[{"source": {"activation": "resid.post"}, "target": ELEMENT_YAML}]),
            "Missing required keys for PatchingElementConfig.*type",
        ),
        (
            _patch_yaml(patch=[{"source": "replace", "target": ELEMENT_YAML}]),
            "Expected a mapping for PatchingElementConfig",
        ),
        ({"method": "exact", "patch": []}, r"Missing required keys for PatchingConfig: \['metric'\]"),
        (
            {"metric": {"logit_diff": ["absolute"]}, "patch": []},
            r"Missing required keys for PatchingConfig: \['method'\]",
        ),
        (_patch_yaml(metric={"accuracy": ["absolute"]}), "accuracy"),
        (_patch_yaml(metric={"logit_diff": ["ratio"]}), "ratio"),
        (_patch_yaml(metric=["logit_diff"]), "Expected a mapping"),
        (_patch_yaml(interpolation_steps=4), "Unknown keys for PatchingConfig.*interpolation_steps"),
    ],
)
def test_a_malformed_patching_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_yaml(PatchingConfig, "broken", raw)


def test_a_run_through_a_cached_model_patches_the_same_way(model: Model, tmp_path: Path) -> None:
    # Patching tokenizes, reads weights and runs with hooks through the wrapper; only its two cached
    # forward passes go through the cache, so the metric must come out exactly as it does unwrapped.
    def patch(target) -> torch.Tensor:
        patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)
        collector = Patcher(target).apply(patching_input, return_collector=True, verbose=False)
        assert collector is not None
        return _metric(collector)

    direct = patch(model)
    cached_model = CachedModel(model, str(tmp_path))

    assert torch.equal(patch(cached_model), direct)
    assert len(list(tmp_path.rglob("*.pt"))) == 2  # one per prompt
    assert torch.equal(patch(cached_model), direct)  # now served from disk


# --------------------------------------------------------------------------------------------------
# The plan a run would execute
# --------------------------------------------------------------------------------------------------
def _every_layer_patch_config() -> PatchingConfig:
    """Replace the residual stream of every layer in turn - one patched run per layer."""
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER],
        type=PatchingType.REPLACE,
    )
    return _patch_config(element, config_id="every_layer_resid_patch")


def test_a_plan_counts_the_patch_runs_and_their_batches(model: Model, monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "planned"
    patching_input = PatchingInput(_every_layer_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)

    forbid_model_runs(monkeypatch, model)
    plan = Patcher(model, batch_size=2).plan(patching_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.steps == model.cfg.n_layers
    assert plan.batches == (model.cfg.n_layers + 1) // 2
    # One patched run per batch, on top of the two cached runs the input prepares.
    assert plan.forward_passes == plan.batches + 2
    assert plan.backward_passes == 0
    assert [(label, prompt) for label, prompt, _ in plan.prompts] == [
        ("source", SOURCE_PROMPT),
        ("target", TARGET_PROMPT),
    ]
    assert f"{plan.steps} patch runs in {plan.batches} batches" in str(plan)


def test_a_plan_counts_a_gradient_runs_passes_instead_of_its_batches(model: Model, monkeypatch, tmp_path: Path) -> None:
    config = replace(
        _every_layer_patch_config(),
        metric=dict.fromkeys((PatchingMetric.LOGIT_DIFF, PatchingMetric.PROBABILITY), [MetricComparison.ABSOLUTE]),
        method=IntegratedGradientsConfig(type="integrated_gradients", interpolation_steps=3),
    )
    patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)

    forbid_model_runs(monkeypatch, model)
    plan = Patcher(model, batch_size=2).plan(patching_input, save_dir=str(tmp_path / "planned"), verbose=False)

    # Every row still gets a score, but they no longer come one batch at a time: each of the two steps
    # short of the clean point interpolates every layer's hook on a pass of its own, the clean point
    # takes one pass for all of them, and each pass goes backward once per metric - on top of the two
    # cached runs.
    passes = 2 * model.cfg.n_layers + 1
    assert plan.steps == model.cfg.n_layers
    assert plan.batches is None
    assert plan.forward_passes == passes + 2
    assert plan.backward_passes == 2 * passes
    assert f"{passes + 2} forward passes, {2 * passes} backward passes" in str(plan)


def test_a_plan_for_a_run_without_a_source_prompt_names_one_prompt(model: Model, monkeypatch, tmp_path: Path) -> None:
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]}
    patching_input = PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona")

    forbid_model_runs(monkeypatch, model)
    plan = Patcher(model).plan(patching_input, save_dir=str(tmp_path / "planned"), verbose=False)

    assert [label for label, _, _ in plan.prompts] == ["target"]
    assert plan.forward_passes == plan.batches + 1


@pytest.mark.parametrize(
    ("source_type", "source_prompt"),
    [
        # A value reads no run at all, so with no source prompt its positions can only be the target's.
        (PatchingType.VALUE_REPLACE, None),
        # Noise corrupts the target run's own activation, so it reads the target's positions however long
        # the source prompt is.
        (PatchingType.GAUSSIAN_NOISE, "The capital of the French Republic is"),
    ],
)
def test_a_source_that_reads_no_source_run_counts_its_positions_in_the_target(
    model: Model, monkeypatch, source_type: PatchingType, source_prompt: str | None
) -> None:
    element = {
        "activation": ModelLayerLocation.RESIDUAL_POST,
        "location": [ModelAxis.LAYER, ModelAxis.POSITION],
        "filter": Filter(inclusion_terms=[{ModelAxis.LAYER: 0, ModelAxis.POSITION: -1}]),
    }
    config = PatchingConfig(
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]},
        method=EXACT,
        patch=[
            SinglePatchingConfig(
                source=PatchingElementConfig(**element, type=source_type),
                target=PatchingElementConfig(**element, type=PatchingType.REPLACE),
            )
        ],
    )
    patching_input = PatchingInput(
        config, source_prompt, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona"
    )

    forbid_model_runs(monkeypatch, model)
    (index,) = Patcher(model).plan(patching_input, verbose=False).index

    last = model.to_tokens(TARGET_PROMPT).shape[1] - 1
    assert index[["source_position", "target_position"]].values.tolist() == [[last, last]]


def test_normalizing_between_two_runs_that_score_the_same_is_rejected_before_any_patched_run(
    model: Model, monkeypatch
) -> None:
    # The same prompt on both sides leaves nothing to normalize against: every row would divide by zero.
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED]}
    patching_input = PatchingInput(
        config, TARGET_PROMPT, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona"
    )
    runs = record_runs(monkeypatch, model)

    with pytest.raises(ValueError, match="normalized_logit_diff divides by"):
        Patcher(model).apply(patching_input, verbose=False)

    # Only the two cached runs were made, both needed to tell.
    assert [run.entry for run in runs] == ["run_with_cache", "run_with_cache"]


def test_a_plan_describes_the_run_that_follows_it(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "patch_run"
    elements = _mlp_mhsa_pattern_elements(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD)
    patching_input = PatchingInput(_patch_config(*elements), SOURCE_PROMPT, TARGET_PROMPT)

    plan = Patcher(model).plan(patching_input, save_dir=str(save_dir), verbose=False)
    # The same input object: planning resolves its indices, the run still caches both prompts.
    collector = Patcher(model).apply(patching_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    # One index per patch, each with its own file.
    assert len(plan.index) == len(elements)
    assert [name for name in plan.outputs if name.startswith("index-")] == [
        f"index-{i}.csv" for i in range(len(elements))
    ]
    for planned, idx in zip(plan.index, patching_input.index, strict=True):
        pd.testing.assert_frame_equal(planned, idx.to_df())
    assert plan.steps == len(collector.result)


# --------------------------------------------------------------------------------------------------
# Visualization - the run's own grid, drawn from result.csv's columns
# --------------------------------------------------------------------------------------------------
_FIRST_THREE_LAYERS = Filter(inclusion_terms=[{ModelAxis.LAYER: {"range": [0, 2]}}])


def _layer_position_config(
    metric: PatchingMetric = PatchingMetric.LOGIT_DIFF,
    index_filter: Filter = _FIRST_THREE_LAYERS,
    comparison: MetricComparison = MetricComparison.ABSOLUTE,
) -> PatchingConfig:
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=index_filter,
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="sweep",
        metric={metric: [comparison]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )


def _draw(
    model: Model,
    config: PatchingConfig,
    save_dir: Path,
    source: str = SOURCE_PROMPT,
    target: str = TARGET_PROMPT,
    **answers,
) -> RenderedFigure:
    Patcher(model).apply(PatchingInput(config, source, target, **answers), save_dir=str(save_dir), verbose=False)
    return rendered_figure(save_dir)


def test_a_patching_run_writes_an_interactive_result_page(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "patch_run"
    figure = _draw(model, _layer_position_config(), save_dir)

    assert "Plotly.newPlot" in (save_dir / "result.html").read_text(encoding="utf-8")
    assert figure.title.startswith("Patching - exact · logit_diff (config: sweep)")


def test_a_gradient_run_names_its_method_on_the_page(model: Model, tmp_path: Path) -> None:
    config = replace(
        _layer_position_config(comparison=MetricComparison.INDIRECT_EFFECT),
        method=PatchingMethodConfig(type="attribution"),
    )

    figure = _draw(model, config, tmp_path / "attributed")

    assert figure.title.startswith("Patching - attribution · indirect_effect_logit_diff (config: sweep)")
    # Nothing was run patched, so no cell has a generation to name.
    assert not any("generation:" in hover for row in figure.trace["hovertext"] for hover in row)


def test_every_column_the_config_names_is_drawn_as_its_own_plot(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "three_columns"
    config = _layer_position_config()
    config.metric = {
        PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE, MetricComparison.NORMALIZED],
        PatchingMetric.KL_DIVERGENCE: [MetricComparison.INDIRECT_EFFECT],
    }

    Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Paris"),
        save_dir=str(save_dir),
        verbose=False,
    )
    figures = rendered_figures(save_dir)
    result = pd.read_csv(save_dir / "result.csv")

    # One plot per column, in the order the config names them, each drawn from its own column.
    assert len(figures) == len(config.metric_columns())
    for figure, column in zip(figures, config.metric_columns(), strict=True):
        assert figure.trace["colorbar"]["title"]["text"] == column
        assert figure.title.startswith(f"Patching - exact · {column} (config: sweep)")
        assert [value for row in figure.trace["z"] for value in row] == pytest.approx(result[column].tolist())


def test_each_plot_is_centred_on_its_own_comparisons_ineffective_value(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "two_comparisons"
    config = _layer_position_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE, MetricComparison.INDIRECT_EFFECT]}
    Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), save_dir=str(save_dir), verbose=False)
    absolute, effect = (figure.trace for figure in rendered_figures(save_dir))

    # The same metric, so the same spread - but the absolute plot sits on the target's own value and
    # the effect plot on zero, because its comparison has already subtracted that value out.
    baseline = float(_unpatched_metrics(model, TARGET_PROMPT)[PatchingMetric.LOGIT_DIFF])
    assert absolute["zmin"] == pytest.approx(2 * baseline - absolute["zmax"])
    assert effect["zmin"] == pytest.approx(-effect["zmax"])


def test_a_sweep_is_drawn_with_positions_across_and_layers_down(model: Model, tmp_path: Path) -> None:
    figure = _draw(model, _layer_position_config(), tmp_path / "patch_run")

    assert figure.kind == "heatmap"
    assert (figure.axis_title("xaxis"), figure.axis_title("yaxis")) == ("position", "layer")
    # plotly draws the first row at the bottom, which is where layer 0 belongs.
    assert figure.ticks("yaxis") == ["0", "1", "2"]
    assert len(figure.trace["z"]) == 3
    assert len(figure.trace["z"][0]) == model.to_tokens(TARGET_PROMPT).shape[1]


def test_the_position_axis_is_labelled_with_the_target_prompts_tokens(model: Model, tmp_path: Path) -> None:
    # The patched run is the target run, so it is the target's tokens that name the positions - the
    # two prompts differ at position 4, which is what makes the choice observable.
    ticks = _draw(model, _layer_position_config(), tmp_path / "patch_run").ticks("xaxis")

    assert ticks[4] == '4: " Spain"'
    assert '4: " France"' not in ticks


def test_the_subtitle_names_what_each_unpatched_prompt_generates(model: Model, tmp_path: Path) -> None:
    source_generation, target_generation = (model.to_single_str_token(token) for token in _generations(model))

    figure = _draw(model, _layer_position_config(), tmp_path / "patch_run")

    assert (
        f'source: "{SOURCE_PROMPT}" generates "{source_generation}" → '
        f'target: "{TARGET_PROMPT}" generates "{target_generation}"'
    ) in figure.title


def test_each_cell_names_what_its_patched_run_generates_on_hover(model: Model, tmp_path: Path) -> None:
    # One layer over every position is drawn as a bar, whose cells are the result's rows in order.
    save_dir = tmp_path / "one_layer"
    config = _layer_position_config(index_filter=Filter(inclusion_terms=[{ModelAxis.LAYER: 0}]))

    figure = _draw(model, config, save_dir)
    generations = pd.read_csv(save_dir / "result.csv", keep_default_na=False)["str_generation"].tolist()

    assert figure.kind == "bar"
    assert len(figure.trace["hovertext"]) == len(generations)
    for hover, generation in zip(figure.trace["hovertext"], generations, strict=True):
        assert hover.split("<br>")[1] == f'generation: "{escape(generation)}"'


def test_the_colour_is_centred_on_the_value_an_ineffective_patch_would_produce(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "patch_run"
    Patcher(model).apply(
        PatchingInput(_layer_position_config(), SOURCE_PROMPT, TARGET_PROMPT), save_dir=str(save_dir), verbose=False
    )

    figure = rendered_figure(save_dir)
    trace = figure.trace
    baseline = float(_unpatched_metrics(model, TARGET_PROMPT)[PatchingMetric.LOGIT_DIFF])

    assert trace["zmin"] == pytest.approx(2 * baseline - trace["zmax"])
    assert trace["colorbar"]["title"]["text"] == "logit_diff"
    # Nothing is printed inside a patching cell; the metric is the colour and the rest is hover.
    assert "text" not in trace


def test_patching_a_prompt_with_itself_leaves_the_scale_intact(model: Model, tmp_path: Path) -> None:
    # Every patch writes back what was already there, so every cell scores the same and the colour
    # range has no spread of its own to take. Reported as an effect, that shared value is zero, which
    # is where the scale is centred.
    figure = _draw(
        model,
        _layer_position_config(comparison=MetricComparison.INDIRECT_EFFECT),
        tmp_path / "no_op",
        source=TARGET_PROMPT,
        correct_token=" Madrid",
        distractor_token=" Barcelona",
    )

    assert {value for row in figure.trace["z"] for value in row} == {0.0}
    assert (figure.trace["zmin"], figure.trace["zmax"]) == (-1.0, 1.0)


def test_an_axis_the_run_does_not_vary_over_is_dropped(model: Model, tmp_path: Path) -> None:
    # One layer over every position: the layer offers no comparison, so it leaves the axes entirely.
    config = _layer_position_config(index_filter=Filter(inclusion_terms=[{ModelAxis.LAYER: 0}]))

    figure = _draw(model, config, tmp_path / "one_layer")

    assert figure.kind == "bar"
    assert figure.axis_title("xaxis") == "position"
    assert figure.axis_title("yaxis") == "logit_diff"
    assert len(figure.trace["y"]) == model.to_tokens(TARGET_PROMPT).shape[1]


def test_a_sparse_index_leaves_holes_in_the_grid(model: Model, tmp_path: Path) -> None:
    # Two correlated points rather than a full grid: the pairs the run never visited stay empty.
    config = _layer_position_config(
        index_filter=Filter(
            inclusion_terms=[
                {ModelAxis.LAYER: 0, ModelAxis.POSITION: 1},
                {ModelAxis.LAYER: 1, ModelAxis.POSITION: 2},
            ]
        )
    )

    figure = _draw(model, config, tmp_path / "sparse")

    assert figure.kind == "heatmap"
    assert [[cell is None for cell in row] for row in figure.trace["z"]] == [[False, True], [True, False]]


def test_explicit_coordinates_are_drawn_as_a_bar_labelled_by_coordinate(model: Model, tmp_path: Path) -> None:
    # An explicit index is a list of points, not a grid, so there is no second axis to lay out.
    coordinates = [
        {ModelAxis.LAYER: -2, ModelAxis.POSITION: -1},
        {ModelAxis.LAYER: -1, ModelAxis.POSITION: -1},
    ]

    figure = _draw(model, _explicit_patch_config(coordinates, coordinates), tmp_path / "coordinates")

    assert figure.kind == "bar"
    assert len(figure.trace["y"]) == 2
    assert figure.ticks("xaxis")[0].startswith("resid.post · layer")


def test_the_axis_falls_back_to_the_target_when_the_source_does_not_span_it(model: Model, tmp_path: Path) -> None:
    # One source coordinate read into two target coordinates: the source is repeated on both rows, so it
    # varies over nothing and cannot name the axis the target laid out.
    source = [{ModelAxis.LAYER: -1, ModelAxis.POSITION: -1}]
    target = [
        {ModelAxis.LAYER: -1, ModelAxis.POSITION: -2},
        {ModelAxis.LAYER: -1, ModelAxis.POSITION: -1},
    ]

    save_dir = tmp_path / "uneven"
    figure = _draw(model, _explicit_patch_config(source, target), save_dir)

    assert pd.read_csv(save_dir / "result.csv")["source_position"].nunique() == 1
    positions = model.to_tokens(TARGET_PROMPT).shape[1]
    assert figure.ticks("xaxis")[0].endswith(f"position {positions - 2}")


def test_a_sweep_written_into_one_location_is_labelled_by_the_source_it_sweeps(model: Model, tmp_path: Path) -> None:
    # Every source row is written into the same target location, so the source is what the grid lays out,
    # and the target, repeated on every row, would give every tick the same label.
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
        filter=_FIRST_THREE_LAYERS,
        type=PatchingType.REPLACE,
    )
    target = _explicit_element([{ModelAxis.LAYER: -1, ModelAxis.POSITION: -1}], ModelLayerLocation.RESIDUAL_POST)
    config = PatchingConfig(
        config_id="into_one",
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )

    figure = _draw(model, config, tmp_path / "into_one")

    tokens = model.to_str_tokens(SOURCE_PROMPT)
    assert figure.ticks("xaxis") == [f'{position}: "{escape(token)}"' for position, token in enumerate(tokens)]
    assert figure.ticks("yaxis") == ["0", "1", "2"]


def test_a_single_coordinate_run_still_writes_a_page(model: Model, tmp_path: Path) -> None:
    # One row has no axis to vary over at all, so the grid degenerates to a single bar.
    coordinates = [{ModelAxis.LAYER: -1, ModelAxis.POSITION: -1}]

    figure = _draw(model, _explicit_patch_config(coordinates, coordinates), tmp_path / "one_coordinate")

    assert figure.kind == "bar"
    assert len(figure.trace["y"]) == 1
    assert figure.title.startswith("Patching - exact · normalized_logit_diff")


def test_a_page_for_a_run_without_a_source_prompt_names_only_the_target(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "ablated"
    config = _last_layer_resid_zero_ablation_config()
    config.metric = {PatchingMetric.LOGIT_DIFF: [MetricComparison.INDIRECT_EFFECT]}
    Patcher(model).apply(
        PatchingInput(config, None, TARGET_PROMPT, correct_token=" Madrid", distractor_token=" Barcelona"),
        save_dir=str(save_dir),
        verbose=False,
    )

    figure = rendered_figure(save_dir)
    target_generation = model.to_single_str_token(_generations(model)[1])

    assert f'target: "{TARGET_PROMPT}" generates "{target_generation}"' in figure.title
    assert "source:" not in figure.title


def test_a_grouped_location_does_not_break_the_page(model: Model, tmp_path: Path) -> None:
    # Grouping puts a list of heads in one index cell; it has to survive being turned into a label.
    element = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelAxis.LAYER, ModelAxis.HEAD],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: -1, ModelAxis.HEAD: [0, 1, 2]}]),
        group=Group({ModelAxis.HEAD: [0, 1]}),
        type=PatchingType.REPLACE,
    )
    config = PatchingConfig(
        metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.ABSOLUTE]},
        method=EXACT,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )

    figure = _draw(model, config, tmp_path / "grouped")

    assert figure.kind == "bar"
    assert any("[0, 1]" in tick for tick in figure.ticks("xaxis"))


def test_the_other_patched_elements_are_named_on_the_page(model: Model, tmp_path: Path) -> None:
    # A companion element patched identically on every run is constant, so it belongs in the subtitle
    # rather than in every cell's hover.
    companion = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelAxis.LAYER, ModelAxis.HEAD],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: 0, ModelAxis.HEAD: 0}]),
        type=PatchingType.REPLACE,
    )
    config = _layer_position_config()
    config.patch.append(SinglePatchingConfig(source=companion, target=companion))

    figure = _draw(model, config, tmp_path / "multi_element")

    assert "also patched: attn.pattern · layer 0 · head 0" in figure.title


def test_the_base_collector_writes_what_its_plan_promises_without_the_page(model: Model, tmp_path: Path) -> None:
    save_dir = tmp_path / "no_viz_plan"
    patching_input = PatchingInput(_layer_position_config(), SOURCE_PROMPT, TARGET_PROMPT)
    patcher = Patcher(model, collector_class=PatchingCollector)

    plan = patcher.plan(patching_input, save_dir=str(save_dir), verbose=False)
    patcher.apply(patching_input, save_dir=str(save_dir), verbose=False)

    # The base collector writes the data and leaves out the page; visualization is opt-in via the subclass.
    assert "result.csv" in plan.outputs
    assert "result.html" not in plan.outputs
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
