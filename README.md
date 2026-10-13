# entoscope

entoscope (Greek *entos*, "within", and *skopein*, "to look") is a framework for mechanistic
interpretability of transformer language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). It reads a model through
four operations:

| Operation | Methods | The question it answers |
|---|---|---|
| [Lens](https://nikolaystanishev.github.io/entoscope/operations/lens.html) | `logit_lens`, `tuned_lens`, `parameter_lens` | What does an activation, or a weight, say about the vocabulary? |
| [Probing](https://nikolaystanishev.github.io/entoscope/operations/probing.html) | `linear` | Where does the model represent a property of its input? |
| [Activation patching](https://nikolaystanishev.github.io/entoscope/operations/patching.html) | `exact`, `attribution`, `integrated_gradients` | Which activations cause the model's behaviour on a task? |
| [Circuit discovery](https://nikolaystanishev.github.io/entoscope/operations/circuit-discovery.html) | `information_flow_routes` | Which routes through the model carry its prediction? |

The four operations share a common design:

- They all have the same shape: a config, an input, an operator and a collector.
- Lens, probing and patching point at the model the same way, naming an activation and the layers,
  heads, neurons and positions to read it at.
- Each run can be built in Python or read from YAML, and can be planned before it is made.
- Each run writes CSV or graph files, and every operation but probing draws an interactive HTML page
  beside them.
- You can register methods of your own, and they run exactly like the built-in ones.

## Install

entoscope needs Python 3.10 to 3.14 and is published on PyPI:

```bash
pip install entoscope
# or, in a uv project
uv add entoscope
```

Intel macOS is limited to transformer-lens 2.15 and Python 3.12 or older; see
[Getting started](https://nikolaystanishev.github.io/entoscope/getting-started.html#install).

## Quickstart

A logit lens over GPT-2's residual stream, after every layer and at every position of a prompt:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = LensConfig(
    method=LensMethodConfig(type="logit_lens"),
    top_k=3,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
    ),
)
Lens(model).apply(LensInput(config, "The capital of France is"), save_dir="runs/logit_lens")
```

Open `runs/logit_lens/result.html` in a browser to see each cell's top token, with layers up and
positions across. `result.csv` holds the same result as a table, and `config.json` the config the run
was made under.

Every operation has a ready-made script and a YAML file of configs on the [Examples](https://nikolaystanishev.github.io/entoscope/examples.html)
page. Saved side by side, a script runs an entry of its YAML file:

```bash
python patching.py --config-id head_sweep
python circuit.py --prompts "The Eiffel Tower is in"
```

## Documentation

The documentation is at <https://nikolaystanishev.github.io/entoscope/>:

- [Getting started](https://nikolaystanishev.github.io/entoscope/getting-started.html) and [Examples](https://nikolaystanishev.github.io/entoscope/examples.html)
- Concepts: [How entoscope is put together](https://nikolaystanishev.github.io/entoscope/concepts/overview.html),
  [Locating activations](https://nikolaystanishev.github.io/entoscope/concepts/locating.html),
  [Configuration files](https://nikolaystanishev.github.io/entoscope/concepts/configuration.html),
  [Running and reading results](https://nikolaystanishev.github.io/entoscope/concepts/running.html),
  [Models and caching](https://nikolaystanishev.github.io/entoscope/concepts/models.html)
- Operations: [Lens](https://nikolaystanishev.github.io/entoscope/operations/lens.html), [Probing](https://nikolaystanishev.github.io/entoscope/operations/probing.html),
  [Activation patching](https://nikolaystanishev.github.io/entoscope/operations/patching.html),
  [Circuit discovery](https://nikolaystanishev.github.io/entoscope/operations/circuit-discovery.html)
- Guides: [Experiment scripts](https://nikolaystanishev.github.io/entoscope/guides/experiment-scripts.html),
  [Custom methods](https://nikolaystanishev.github.io/entoscope/guides/custom-methods.html), [Datasets](https://nikolaystanishev.github.io/entoscope/guides/datasets.html)
- [API reference](https://nikolaystanishev.github.io/entoscope/reference/operation.html), generated from the docstrings
- [Development](https://nikolaystanishev.github.io/entoscope/development.html)

## Attribution and licensing

entoscope is released under the MIT License, except for one vendored file.
The Information Flow Routes method reuses the contribution-graph math of the
[LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool), vendored as
`src/entoscope/circuit/vendor/contributions.py` (see `src/entoscope/circuit/vendor/NOTICE`), which is licensed
**CC BY-NC 4.0**. Because that code is non-commercial, the Circuit Discovery `information_flow_routes`
method, and any results produced with it, may be used for **non-commercial purposes only**. Method
reference:

> Javier Ferrando and Elena Voita. "Information Flow Routes: Automatically Interpreting Language
> Models at Scale." 2024. https://arxiv.org/abs/2403.00824
