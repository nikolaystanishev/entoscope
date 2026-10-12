# intex

intex is a framework for mechanistic interpretability of transformer language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). It reads a model through
four operations, each with its own methods:

| Operation | Methods | The question it answers |
|---|---|---|
| [Lens](operations/lens.md) | `logit_lens`, `tuned_lens`, `parameter_lens` | What does an activation, or a weight, say about the vocabulary? |
| [Probing](operations/probing.md) | `linear` | Where does the model represent a property of its input? |
| [Activation patching](operations/patching.md) | `exact`, `attribution`, `integrated_gradients` | Which activations cause the model's behaviour on a task? |
| [Circuit discovery](operations/circuit-discovery.md) | `information_flow_routes` | Which routes through the model carry its prediction? |

The operations share most of what you need to learn about them:

- Every operation has the same four parts: a config says what to run, an input pairs it with the
  prompts, an operator runs it, and a collector writes the results.
- Lens, probing and patching name the activations they read with the same locator: an activation,
  plus the layers, heads, neurons and positions to read it at.
- A config can be built in Python or read from a YAML file, and every run saves the config it ran
  under next to its results.
- `plan` describes a run (what it reads, how many forward passes, which files) without running the
  model.
- Results are CSV files (graph files for circuits), and every operation but probing draws them on
  an interactive HTML page you open in a browser.
- Every method is a registered class, and your own methods register and run the same way.

## Where to go

- [Getting started](getting-started.md): install intex and make a first run.
- Concepts:
  - [How intex is put together](concepts/overview.md)
  - [Locating activations](concepts/locating.md)
  - [Configuration files](concepts/configuration.md)
  - [Running and reading results](concepts/running.md)
  - [Models and caching](concepts/models.md)
- Operations: [Lens](operations/lens.md), [Probing](operations/probing.md),
  [Activation patching](operations/patching.md), [Circuit discovery](operations/circuit-discovery.md).
- Guides: [Experiment scripts](guides/experiment-scripts.md), [Custom methods](guides/custom-methods.md),
  [Datasets](guides/datasets.md).
- [API reference](reference/operation.md), generated from the docstrings.
- [Development](development.md): working on intex itself.

The repository's
[`examples/`](https://github.com/nikolaystanishev/intex/tree/main/examples) directory has a runnable
script and a YAML file of configs for every operation.

```{toctree}
:hidden:
:caption: Start

getting-started
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
operations/probing
operations/patching
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
reference/probe
reference/patching
reference/circuit
reference/dataset
reference/script
```

```{toctree}
:hidden:
:caption: Project

development
```
