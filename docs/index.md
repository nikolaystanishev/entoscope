# entoscope

entoscope is a modular framework for mechanistic interpretability: it runs operations that analyze
the internals of large language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). It supports four operations
for now, each computed by one or more methods:

| Operation | Methods | Description |
|---|---|---|
| [Lens](operations/lens.md) | `logit_lens`, `tuned_lens`, `parameter_lens` | Projects activations and weights onto the vocabulary to show which tokens they promote. |
| [Activation patching](operations/patching.md) | `exact`, `attribution`, `integrated_gradients` | Patches activations from one run into another and measures how the output changes. |
| [Probing](operations/probing.md) | `linear` | Trains classifiers on activations to find where the model represents a property of its input. |
| [Circuit discovery](operations/circuit-discovery.md) | `information_flow_routes` | Traces the routes through attention heads and MLPs that carry the model's prediction. |

The operations share most of what you need to learn about them:

- Every operation has the same four parts: a config says what to run, an input pairs it with the
  prompts, an operator runs it, and a collector writes the results.
- Where an operation works in the model is specified flexibly: any activation, swept over layers,
  heads, neurons and positions, narrowed by filters, grouped to read several together, or listed
  point by point.
- A config can be built in Python or read from a YAML file, and every run saves the config it ran
  under next to its results.
- `plan` describes a run (what it reads, how many forward passes, which files) without running the
  model.
- Results are CSV files (graph files for circuits), and every operation but probing draws them on
  an interactive HTML page you open in a browser.
- Every method is a registered class, and your own methods register and run the same way.

## Where to go

- [Getting started](getting-started.md): install entoscope and make a first run.
- [Examples](examples.md): a script and a YAML file of configs for every operation.
- Concepts:
  - [How entoscope is put together](concepts/overview.md)
  - [Locating activations](concepts/locating.md)
  - [Configuration files](concepts/configuration.md)
  - [Running and reading results](concepts/running.md)
  - [Models and caching](concepts/models.md)
- Operations: [Lens](operations/lens.md), [Activation patching](operations/patching.md),
  [Probing](operations/probing.md), [Circuit discovery](operations/circuit-discovery.md).
- Guides: [Experiment scripts](guides/experiment-scripts.md), [Custom methods](guides/custom-methods.md),
  [Datasets](guides/datasets.md).
- [API reference](reference/operation.md), generated from the docstrings.
- [Development](development.md): working on entoscope itself.

```{toctree}
:hidden:
:caption: Start

getting-started
examples
```

```{toctree}
:hidden:
:caption: Concepts

concepts/overview
concepts/locating
concepts/configuration
concepts/running
concepts/models
```

```{toctree}
:hidden:
:caption: Operations

operations/lens
operations/patching
operations/probing
operations/circuit-discovery
```

```{toctree}
:hidden:
:caption: Guides

guides/experiment-scripts
guides/custom-methods
guides/datasets
```

```{toctree}
:hidden:
:caption: API reference

reference/operation
reference/model
reference/lens
reference/patching
reference/probe
reference/circuit
reference/dataset
reference/script
```

```{toctree}
:hidden:
:caption: Project

development
```
