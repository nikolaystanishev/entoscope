# entoscope

entoscope (Greek *entos*, "within", and *skopein*, "to look") is a framework for mechanistic
interpretability of transformer language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). It reads a model through
four operations:

| Operation | Methods | The question it answers |
|---|---|---|
| [Lens](docs/operations/lens.md) | `logit_lens`, `tuned_lens`, `parameter_lens` | What does an activation, or a weight, say about the vocabulary? |
| [Probing](docs/operations/probing.md) | `linear` | Where does the model represent a property of its input? |
| [Activation patching](docs/operations/patching.md) | `exact`, `attribution`, `integrated_gradients` | Which activations cause the model's behaviour on a task? |
| [Circuit discovery](docs/operations/circuit-discovery.md) | `information_flow_routes` | Which routes through the model carry its prediction? |

The four operations share a common design:

- They all have the same shape: a config, an input, an operator and a collector.
- Lens, probing and patching point at the model the same way, naming an activation and the layers,
  heads, neurons and positions to read it at.
- Each run can be built in Python or read from YAML, and can be planned before it is made.
- Each run writes CSV or graph files, and every operation but probing draws an interactive HTML page
  beside them.
- You can register methods of your own, and they run exactly like the built-in ones.

## Install

entoscope needs Python 3.10 to 3.14 and is installed from this repository:

```bash
git clone git@github.com:nikolaystanishev/entoscope.git
cd entoscope
uv sync
```

or, from another project, `uv add git+ssh://git@github.com/nikolaystanishev/entoscope.git`. Intel macOS is
limited to transformer-lens 2.15 and Python 3.12 or older; see
[Getting started](docs/getting-started.md#install).

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

Every operation has a ready-made script and a YAML file of configs in [`examples/`](examples):

```bash
uv run python examples/patching.py --config-id head_sweep
uv run python examples/circuit.py --prompts "The Eiffel Tower is in"
```

## Documentation

The docs live in [`docs/`](docs) and build into a site with
`uv run --group docs sphinx-build docs docs/_build/html`:

- [Getting started](docs/getting-started.md)
- Concepts: [How entoscope is put together](docs/concepts/overview.md),
  [Locating activations](docs/concepts/locating.md), [Configuration files](docs/concepts/configuration.md),
  [Running and reading results](docs/concepts/running.md), [Models and caching](docs/concepts/models.md)
- Operations: [Lens](docs/operations/lens.md), [Probing](docs/operations/probing.md),
  [Activation patching](docs/operations/patching.md), [Circuit discovery](docs/operations/circuit-discovery.md)
- Guides: [Experiment scripts](docs/guides/experiment-scripts.md),
  [Custom methods](docs/guides/custom-methods.md), [Datasets](docs/guides/datasets.md)
- [Development](docs/development.md); the architecture is described in [AGENTS.md](AGENTS.md)

The site also has an API reference generated from the docstrings.

## Attribution and licensing

The Information Flow Routes method reuses the contribution-graph math of the
[LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool), vendored as
`src/entoscope/circuit/vendor/contributions.py` (see `src/entoscope/circuit/vendor/NOTICE`), which is licensed
**CC BY-NC 4.0**. Because that code is non-commercial, the Circuit Discovery `information_flow_routes`
method, and any results produced with it, may be used for **non-commercial purposes only**. Method
reference:

> Javier Ferrando and Elena Voita. "Information Flow Routes: Automatically Interpreting Language
> Models at Scale." 2024. https://arxiv.org/abs/2403.00824
