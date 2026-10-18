# entoscope

[![PyPI](https://img.shields.io/pypi/v/entoscope)](https://pypi.org/project/entoscope/)
[![Python](https://img.shields.io/pypi/pyversions/entoscope)](https://pypi.org/project/entoscope/)
[![CI](https://github.com/nikolaystanishev/entoscope/actions/workflows/ci.yml/badge.svg)](https://github.com/nikolaystanishev/entoscope/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://nikolaystanishev.github.io/entoscope/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/nikolaystanishev/entoscope/blob/main/LICENSE)

entoscope is a modular framework for mechanistic interpretability. It analyzes the internals of large
language models through a set of operations that share one design: point an operation at any
activation of the model, at the layers, heads, neurons and positions you choose, and get the result
as a table and an interactive page. It is built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Operations

entoscope supports four operations for now, each computed by one or more methods:

| Operation | Methods | Description |
|---|---|---|
| [Lens](https://nikolaystanishev.github.io/entoscope/operations/lens.html) | `logit_lens`, `tuned_lens`, `parameter_lens` | Projects activations and weights onto the vocabulary to show which tokens they promote. |
| [Activation patching](https://nikolaystanishev.github.io/entoscope/operations/patching.html) | `exact`, `attribution`, `integrated_gradients` | Patches activations from one run into another and measures how the output changes. |
| [Probing](https://nikolaystanishev.github.io/entoscope/operations/probing.html) | `linear` | Trains classifiers on activations to find where the model represents a property of its input. |
| [Circuit discovery](https://nikolaystanishev.github.io/entoscope/operations/circuit-discovery.html) | `information_flow_routes` | Traces the routes through attention heads and MLPs that carry the model's prediction. |

## Features

- **One design for every operation.** A config says what to run, an input adds the prompts, an
  operator runs it and a collector writes the results. Once one operation makes sense, so do the rest.
- **Flexible locating.** Read any activation, swept over layers, heads, neurons and positions,
  narrowed by filters, grouped to read several together, or listed point by point.
- **Python or YAML.** Build a config in code, or keep every config of an experiment in one YAML file.
  Each run saves the config it ran under beside its results.
- **Plan before you run.** `plan()` reports the rows, forward passes and files of a run without
  running the model.
- **Results you can open.** Every run writes CSV tables, and every operation but probing draws an
  interactive HTML page that opens straight from disk.
- **Your own methods.** Methods are registered by name, and one you register runs through the same
  operator, plan and result page as the built-in ones.
- **Cached activations.** `CachedModel` keeps a model's activations on disk between runs.

## Installation

entoscope depends on Python 3.11 to 3.14 and is published on PyPI:

```bash
pip install entoscope
```

> [!NOTE]
> On Intel macOS, PyTorch publishes no wheels after 2.2, so entoscope installs with transformer-lens
> 2.15 and Python 3.12 or older there. See
> [Getting started](https://nikolaystanishev.github.io/entoscope/getting-started.html#install).

## Quickstart

A logit lens reads GPT-2's residual stream after every layer, at every position of a prompt, as if the
model stopped there and predicted the next token:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import (
    Lens,
    LensConfig,
    LensInput,
    LensMethodConfig,
    ModelAxis,
    ModelLayerLocation,
    ModelLocatorConfig,
)

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = LensConfig(
    method=LensMethodConfig(type="logit_lens"),
    top_k=3,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
    ),
)
Lens(model).apply(LensInput(config, "The capital of France is"), save_dir="runs/logit_lens")
```

The run writes four files into `runs/logit_lens/`:

- `result.html`: an interactive grid, layers up and positions across, with each cell's top tokens.
- `result.csv`: the same result as a table, one row per layer and position.
- `index.csv`: the rows the run swept.
- `config.json`: the config the run was made under.

## Running from a config file

The same operators run from YAML. This entry of
[`examples/patching.yaml`](https://github.com/nikolaystanishev/entoscope/blob/main/examples/patching.yaml)
patches the residual stream of a clean prompt into a corrupted one, at every layer and position:

```yaml
residual_sweep:
  metric:
    logit_diff: [normalized]
    kl_divergence: [indirect_effect]
  method: exact
  patch:
    - source: &residual
        activation: resid.pre
        location: [layer, position]
        type: replace
      target: *residual
```

```python
from entoscope import ConfigFactory, Patcher, PatchingConfig, PatchingInput

config = ConfigFactory[PatchingConfig]("examples/patching.yaml").create("residual_sweep")
patching_input = PatchingInput(
    config,
    "The capital of France is",  # source
    "The capital of Spain is",  # target
    correct_token=" Paris",
    distractor_token=" Madrid",
)
collector = Patcher(model).apply(patching_input, save_dir="runs/residual_sweep", return_collector=True)

grid = collector.result.pivot(index="target_layer", columns="target_position", values="normalized_logit_diff")
print(grid.round(2))
```

`normalized_logit_diff` is 0 where a patch changed nothing and 1 where it restored the clean answer.
The grid shows the answer moving through GPT-2: up to layer 9 it sits at the `" France"` position, and
by the last layers it has moved to the final position:

```text
target_position    0    1    2    3     4     5
target_layer
0                0.0  0.0  0.0  0.0  1.00  0.00
1                0.0  0.0  0.0  0.0  1.00  0.00
2                0.0  0.0  0.0  0.0  1.00 -0.00
3                0.0  0.0  0.0  0.0  1.00  0.01
4                0.0  0.0  0.0  0.0  1.00  0.01
5                0.0  0.0  0.0  0.0  1.02  0.03
6                0.0  0.0  0.0  0.0  1.02  0.04
7                0.0  0.0  0.0  0.0  0.98  0.08
8                0.0  0.0  0.0  0.0  0.98  0.09
9                0.0  0.0  0.0  0.0  0.90  0.15
10               0.0  0.0  0.0  0.0  0.30  0.65
11               0.0  0.0  0.0  0.0  0.13  0.80
```

Every operation has a YAML file of configs like this one and a script that runs its entries from the
command line, on the [Examples](https://nikolaystanishev.github.io/entoscope/examples.html) page.

## Documentation

The documentation is at <https://nikolaystanishev.github.io/entoscope/>:

- [Getting started](https://nikolaystanishev.github.io/entoscope/getting-started.html): install and make a first run.
- [Concepts](https://nikolaystanishev.github.io/entoscope/concepts/overview.html): how operations are
  put together, locating activations, config files, reading results, models and caching.
- [Operations](https://nikolaystanishev.github.io/entoscope/operations/lens.html): every operation,
  its methods and the files it writes.
- [Guides](https://nikolaystanishev.github.io/entoscope/guides/custom-methods.html): experiment
  scripts, custom methods and datasets.
- [API reference](https://nikolaystanishev.github.io/entoscope/reference/operation.html).

## Contributing

Issues and pull requests are welcome. [Development](https://nikolaystanishev.github.io/entoscope/development.html)
covers setting up the repository, running the tests and building the docs.

## License

entoscope is released under the [MIT License](https://github.com/nikolaystanishev/entoscope/blob/main/LICENSE),
except for one vendored file. Circuit discovery's `information_flow_routes` method reuses the
contribution-graph math of the [LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool),
vendored as `src/entoscope/circuit/vendor/contributions.py` under **CC BY-NC 4.0** (see its
[NOTICE](https://github.com/nikolaystanishev/entoscope/blob/main/src/entoscope/circuit/vendor/NOTICE)).
That method, and any results produced with it, may be used for **non-commercial purposes only**.
The method is described in:

> Javier Ferrando and Elena Voita. "Information Flow Routes: Automatically Interpreting Language
> Models at Scale." 2024. https://arxiv.org/abs/2403.00824
