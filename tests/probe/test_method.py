import pytest
import torch

from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.probe.config import PoolingType, ProbeConfig, ProbeMetric, ProbeType
from intex.probe.method.factory import ProbeMethodFactory
from intex.probe.method.linear import LinearProbe
from intex.probe.method.probe import AbstractProbeMethod


def _config(**overrides) -> ProbeConfig:
    return ProbeConfig(
        type=ProbeType.LINEAR,
        pool={ModelLocation.POSITION: PoolingType.MEAN},
        metric=[ProbeMetric.ACCURACY],
        source=ModelLocatorConfig(activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER]),
        **overrides,
    )


def test_abstract_method_not_instantiable():
    with pytest.raises(TypeError):
        AbstractProbeMethod(object(), _config())


def test_factory_get():
    assert ProbeMethodFactory.get(ProbeType.LINEAR) is LinearProbe


def test_factory_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown probe type"):
        ProbeMethodFactory.get("mlp")


def test_factory_create_wires_model_and_config():
    model = object()
    config = _config(max_iter=250, seed=3)
    method = ProbeMethodFactory.create(config, model)
    assert isinstance(method, LinearProbe)
    assert method.model is model
    assert method.config is config
    assert method.probe.max_iter == 250
    assert method.probe.random_state == 3


def test_linear_probe_fits_separable_data():
    features = torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    method = ProbeMethodFactory.create(_config(), object())
    method.fit(features, labels)

    assert method.predict(features).tolist() == labels.tolist()
    probabilities = method.predict_proba(features)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities[0] < probabilities[-1]
