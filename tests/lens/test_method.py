import pytest
import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import LogitLens
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement


def _config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )


def _element(config: LensConfig) -> IndexElement[ModelLocatorConfig]:
    return IndexElement(config.source, ["activation", "layer", "position"], [ModelLayerLocation.RESIDUAL_POST, 0, 0])


def test_abstract_method_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractLensMethod(object(), _config())


def test_factory_get_returns_logit_lens():
    assert LensMethodFactory.get(LensType.LOGIT_LENS) is LogitLens


def test_factory_get_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown lens type"):
        LensMethodFactory.get("tuned_lens")


def test_factory_create_builds_configured_instance():
    model = object()
    config = _config()

    method = LensMethodFactory.create(config, model)

    assert isinstance(method, LogitLens)
    assert method.model is model
    assert method.config is config


def test_logit_lens_project_is_not_implemented():
    config = _config()
    method = LogitLens(object(), config)

    with pytest.raises(NotImplementedError):
        method.project(torch.zeros(1, 4), _element(config))
