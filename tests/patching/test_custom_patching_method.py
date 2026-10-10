"""A patching method a user registers, run end to end the way the built-in ones are."""

import json
from dataclasses import dataclass

import pytest
import torch

from intex.patching.config import PatchingConfig, PatchingMethodConfig
from intex.patching.input import PatchingInput
from intex.patching.method.factory import PatchingMethodFactory
from intex.patching.method.patching import AbstractPatchingMethod, RowResult
from intex.patching.patcher import Patcher
from intex.util.collections import zip_fill

from ..conftest import forbid_model_runs, rendered_figures

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"
CORRECT_TOKEN, DISTRACTOR_TOKEN = " Paris", " Madrid"
DELTA = 0.5

ELEMENT_YAML = {"activation": "resid.post", "location": ["layer"], "type": "replace"}
CONSTANT_YAML = {
    "method": {"type": "constant", "delta": DELTA},
    "metric": {"logit_diff": ["absolute", "indirect_effect"], "kl_divergence": ["indirect_effect"]},
    "patch": [{"source": ELEMENT_YAML, "target": ELEMENT_YAML}],
}


@dataclass(kw_only=True)
class ConstantConfig(PatchingMethodConfig):
    delta: float = 0.0


class ConstantMethod(AbstractPatchingMethod[ConstantConfig]):
    """Reports the same change in every metric for every row, without running anything."""

    def scores(self, patching_input, verbose=True):
        delta = torch.tensor(self.method_config.delta)
        for group in zip_fill(*patching_input.index):
            yield RowResult(group[0].output, self._report(patching_input, dict.fromkeys(self.config.metric, delta)))

    def passes(self, runs, batches):
        return 0, 0


@pytest.fixture
def constant_method(registries):
    return PatchingMethodFactory.register("constant")(ConstantMethod)


def _input(config: PatchingConfig) -> PatchingInput:
    return PatchingInput(
        config, SOURCE_PROMPT, TARGET_PROMPT, correct_token=CORRECT_TOKEN, distractor_token=DISTRACTOR_TOKEN
    )


def _target_logit_diff(model) -> float:
    with torch.no_grad():
        logits = model(TARGET_PROMPT)[0, -1]
    return float(logits[model.to_single_token(CORRECT_TOKEN)] - logits[model.to_single_token(DISTRACTOR_TOKEN)])


def test_a_registered_method_is_reported_under_every_comparison_the_config_names(
    model, config_from_yaml, constant_method
):
    config = config_from_yaml(PatchingConfig, "constant", CONSTANT_YAML)

    result = Patcher(model).apply(_input(config), return_collector=True, verbose=False).result

    rows = model.cfg.n_layers
    assert len(result) == rows
    # The change rides on the unpatched target run's own metric, and an effect reads it as it is -
    # except a divergence's, whose effect is its drop.
    assert result["logit_diff"].tolist() == pytest.approx([_target_logit_diff(model) + DELTA] * rows, rel=1e-4)
    assert result["indirect_effect_logit_diff"].tolist() == pytest.approx([DELTA] * rows)
    assert result["indirect_effect_kl_divergence"].tolist() == pytest.approx([-DELTA] * rows)
    # Nothing was run patched, so no row names a generation.
    assert "generation" not in result.columns


def test_a_plan_counts_the_registered_methods_own_passes(model, monkeypatch, config_from_yaml, constant_method):
    forbid_model_runs(monkeypatch, model)

    plan = Patcher(model, batch_size=2).plan(
        _input(config_from_yaml(PatchingConfig, "constant", CONSTANT_YAML)), verbose=False
    )

    assert plan.describe().splitlines()[:3] == ["Patcher plan", "  method: constant", "  config: constant"]
    assert plan.steps == model.cfg.n_layers
    # It runs no patched model, so it has no batches - only the two runs the input caches.
    assert plan.batches is None
    assert (plan.forward_passes, plan.backward_passes) == (2, 0)


def test_the_page_names_the_registered_method(model, config_from_yaml, constant_method, tmp_path):
    save_dir = tmp_path / "constant"

    Patcher(model).apply(
        _input(config_from_yaml(PatchingConfig, "constant", CONSTANT_YAML)), save_dir=str(save_dir), verbose=False
    )

    titles = [figure.title for figure in rendered_figures(save_dir)]
    assert len(titles) == 3
    assert all(title.startswith("Patching - constant · ") for title in titles)
    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["method"] == {"type": "constant", "delta": DELTA}
