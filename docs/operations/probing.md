# Probing

Probing asks where a model represents a property of its input. Every location in the index gets a
classifier, trained to tell the activations of positive example prompts from those of negative ones
and scored by the metrics you name.

| Method | Method config | Fields |
|---|---|---|
| `linear` | `LinearProbeConfig` | `max_iter` (default `1000`), `seed` (default `42`) |

## A run

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from entoscope.probe.config import PoolingType, ProbeConfig, ProbeMetric
from entoscope.probe.input import ProbeInput
from entoscope.probe.method.linear import LinearProbeConfig
from entoscope.probe.prober import Prober

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

# A sentiment probe on the last token's residual stream, after every layer.
config = ProbeConfig(
    method=LinearProbeConfig(type="linear"),
    pool={ModelLocation.POSITION: PoolingType.LAST},
    metric=[ProbeMetric.ACCURACY, ProbeMetric.AUROC],
    source=ModelLocatorConfig(activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER]),
)
positive = ["The movie was absolutely wonderful", "I loved every minute of it", "What a delightful surprise"]
negative = ["The movie was dreadful and boring", "I hated every minute of it", "What a miserable waste of time"]

collector = Prober(model).apply(
    ProbeInput(config, positive, negative), save_dir="runs/sentiment", return_collector=True
)
print(collector.result)  # one row per layer, with an accuracy and an auroc column
```

[`examples/probe.yaml`](https://github.com/nikolaystanishev/entoscope/blob/main/examples/probe.yaml) has
more configs, and [`examples/probe.py`](https://github.com/nikolaystanishev/entoscope/blob/main/examples/probe.py)
a larger set of example prompts.

### Config and input

`ProbeConfig` fields:

| Field | Meaning | Default |
|---|---|---|
| `method` | the probe, as its method config | required |
| `pool` | the axes to reduce, each to a pooling: `mean`, `max`, `sum` or `last` | none |
| `metric` | the scores to report: `accuracy`, `precision`, `recall`, `f1`, `auroc` | required |
| `source` | the locations to probe, a [locator](../concepts/locating.md) | required |

`ProbeInput(config, positive_prompts, negative_prompts)` labels the positive prompts 1 and the
negative ones 0; both lists must be non-empty. The prompts may tokenize to different lengths, and then
positions resolve against each prompt separately (see
[Prompts of different lengths](../concepts/locating.md#prompts-of-different-lengths)).

## What a probe sees

For each row of the index, every prompt contributes one feature vector:

1. The row's slice of the activation is taken. Axes the row pins to a single value drop out, while
   axes it leaves whole, or names with a list, stay.
2. Each remaining axis named in `pool` is reduced: `mean`, `max` and `sum` over its entries, or `last`
   to keep its final entry, which for `position` is the prompt's last token.
3. Whatever remains is flattened into the feature vector.

For example, `resid.post` located by `[layer]` and pooled over `position` gives `d_model` features per
prompt. `attn.z` located by `[layer, head]` and pooled over `position` gives `d_head` features per
head. Only `position` and `head` can be pooled: `layer` is never an axis of the tensor, and `neuron` is
the feature axis itself. A `pool` entry for an axis the row has already pinned is ignored. For
`attn.pattern`, pooling `position` reduces both the query and the key positions.

The tokenizer prepends a BOS token, so `mean` and `sum` over positions include position 0. That
position is the same for every prompt, since it attends only to itself. In GPT-2 its residual stream
is also far larger than the others: for the first prompt above, about 3100 in norm after layer 6,
against about 100 at the other positions. `last` pooling, or a pinned position such as `-1`, leaves
it out.

## Training and scoring

Every row trains a probe of its own, built fresh from the method config. The `linear` probe is
scikit-learn's `LogisticRegression` with its default L2 regularization, `max_iter` iterations at most
and `seed` as its random state.

A probe is scored on the same examples it was trained on: there is no held-out split. A handful of
examples in hundreds of dimensions separate perfectly at almost every location, which makes every
score 1.0. Read the scores as how separable the classes are at a location, use many examples, and
evaluate generalization on prompts the probe never saw.

| Metric | Computed from |
|---|---|
| `accuracy`, `precision`, `recall`, `f1` | the probe's 0/1 predictions; precision, recall and F1 are 0 where undefined |
| `auroc` | the probe's positive-class probabilities |

## Output

`result.csv` has one row per location: the location, then a column per metric, named as the metric.
`probes.pkl` holds the fitted probes, pickled as a dict from result row to estimator:

```python
import pickle

with open("runs/sentiment/probes.pkl", "rb") as file:
    probes = pickle.load(file)
print(probes[11].coef_.shape)  # the last layer's probe: (1, 768)
```

Probing draws no page.

## References

- Guillaume Alain and Yoshua Bengio. [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644). 2016.
- Yonatan Belinkov. [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452). 2021.
