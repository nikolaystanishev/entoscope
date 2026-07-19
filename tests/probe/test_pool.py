import pytest
import torch

from intex.model.config import ModelLocation
from intex.probe.config import PoolingType
from intex.probe.pool import (
    AbstractPooling,
    LastPooling,
    MaxPooling,
    MeanPooling,
    SumPooling,
    pool,
)

POS_HEAD_AXES = [ModelLocation.POSITION, ModelLocation.HEAD, ModelLocation.NEURON]


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        (PoolingType.MEAN, MeanPooling),
        (PoolingType.MAX, MaxPooling),
        (PoolingType.SUM, SumPooling),
        (PoolingType.LAST, LastPooling),
    ],
)
def test_get(pooling, expected):
    assert AbstractPooling.get(pooling) is expected


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pooling type"):
        AbstractPooling.get("median")


@pytest.mark.parametrize(
    ("pooling", "reduce"),
    [
        (PoolingType.MEAN, lambda activation: activation.mean(dim=0)),
        (PoolingType.MAX, lambda activation: activation.amax(dim=0)),
        (PoolingType.SUM, lambda activation: activation.sum(dim=0)),
        (PoolingType.LAST, lambda activation: activation[-1]),
    ],
)
def test_single_axis_reductions(pooling, reduce):
    activation = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    axes = [ModelLocation.POSITION, ModelLocation.NEURON]
    assert torch.equal(pool(activation, axes, {ModelLocation.POSITION: pooling}), reduce(activation))


def test_unlisted_axes_flatten_into_features():
    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pooled = pool(activation, POS_HEAD_AXES, {ModelLocation.POSITION: PoolingType.MEAN})
    assert torch.equal(pooled, activation.mean(dim=0).flatten())


def test_empty_config_flattens_everything():
    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(pool(activation, POS_HEAD_AXES, {}), activation.flatten())


def test_mixed_types_reduce_in_axis_order():
    # Position is reduced before head regardless of config order; the reversed order would give 9.5.
    activation = torch.tensor([[[0.0], [10.0]], [[9.0], [5.0]]])
    config = {ModelLocation.HEAD: PoolingType.MAX, ModelLocation.POSITION: PoolingType.MEAN}
    assert torch.equal(pool(activation, POS_HEAD_AXES, config), torch.tensor([7.5]))


def test_position_matches_both_pattern_axes():
    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pooled = pool(activation, [ModelLocation.HEAD, "dest_pos", "src_pos"], {ModelLocation.POSITION: PoolingType.SUM})
    assert torch.equal(pooled, activation.sum(dim=1).sum(dim=1))


def test_non_matching_keys_ignored():
    activation = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    axes = [ModelLocation.POSITION, ModelLocation.NEURON]
    config = {ModelLocation.POSITION: PoolingType.MEAN, ModelLocation.HEAD: PoolingType.MAX}
    assert torch.equal(pool(activation, axes, config), activation.mean(dim=0))


def test_one_dimensional_flattens_or_reduces():
    activation = torch.arange(4, dtype=torch.float32)
    assert torch.equal(pool(activation, [ModelLocation.NEURON], {}), activation)
    pooled = pool(activation, [ModelLocation.POSITION], {ModelLocation.POSITION: PoolingType.MEAN})
    assert torch.equal(pooled, activation.mean().reshape(1))


def test_zero_dimensional_becomes_single_feature():
    activation = torch.tensor(3.5)
    assert torch.equal(pool(activation, [], {}), torch.tensor([3.5]))
