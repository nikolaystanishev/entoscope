# How entoscope is put together

Every operation is made of the same four parts, so once one operation makes sense the others mostly
do too.

| Part | What it is | Lens | Probing | Activation patching | Circuit discovery |
|---|---|---|---|---|---|
| Config | What to run: a dataclass, built in code or read from YAML | `LensConfig` | `ProbeConfig` | `PatchingConfig` | `CircuitConfig` |
| Input | The config together with the prompts it runs on | `LensInput` | `ProbeInput` | `PatchingInput` | `CircuitInput` |
| Operator | Runs the config's method over the input | `Lens` | `Prober` | `Patcher` | `CircuitDiscoverer` |
| Collector | Gathers one result per row and writes the run's files | `LensVisualizationCollector` | `ProbeCollector` | `PatchingVisualizationCollector` | `CircuitVisualizationCollector` |

The configs, inputs and operators, with the method configs, the locator classes and `CachedModel`, are
importable from `entoscope` itself: `from entoscope import Lens, LensConfig, LensInput`. Each operation's
package - `entoscope.lens`, `entoscope.probe`, `entoscope.patching` and `entoscope.circuit` - exports the
rest of its public classes, such as its collectors, its methods and its method factory, and
`entoscope.model` and `entoscope.operation` export the shared ones. Each class is defined in a module
named after its part (`entoscope.lens.config`, `entoscope.lens.input`, `entoscope.lens.lens`,
`entoscope.lens.collector`, ...), which is how the [API reference](../reference/operation.md) is laid out.

## A run, step by step

`operator.apply(input, save_dir)` does the same thing for every operation:

1. It builds the method the config's `method` names. A method that cannot be built for this
   config fails here, before any work is done.
2. It prepares the input: tokenizes the prompts, resolves the index (below) and runs the model,
   caching only the activations the index reads.
3. It runs the method over every row of the index, handing each row's result to the collector.
4. The collector writes the results into `save_dir`, with the config the run was made under.

`operator.plan(input, save_dir)` stops after resolving the index and reports what the rest would do.
See [Running and reading results](running.md).

## The index

Lens, probing and patching read activations at many locations: every layer, every head of a layer,
one neuron at the last position, and so on. The config's locator (`source`, and `target` for
patching) names those locations, and resolving it against the model and the prompt gives the
**index**: a table with one row per location.

The index drives everything after it. The method runs once per row, the result has one row per row of
the index, and `index.csv` holds the index itself. The result page lays the rows out on the axes the
index varies over. [Locating activations](locating.md) describes the locator.

Circuit discovery is the exception. A circuit spans every layer, position and head, so its config has
no locator and its run makes one graph per prompt.

## Methods

An operation can be computed more than one way: a lens can read the residual stream directly or
through trained translators, and patching can run the patched model or estimate the effect from
gradients. The config's `method` field picks the way. It is a method config whose `type` names the
method, and whose other fields are that method's settings:

```python
from entoscope import ParameterLensConfig

method = ParameterLensConfig(type="parameter_lens", scale=-1.0)
```

Every operation keeps a registry of its methods by name. The built-in methods are registered the same
way as one you write, so your own method runs through the same operator, collector, plan and result
page. See [Custom methods](../guides/custom-methods.md).

## Collectors and pages

The collector decides what a run leaves behind. Each operator's default collector also draws an
interactive HTML page; passing the base collector writes the data files alone:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import Lens
from entoscope.lens import LensCollector

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

lens = Lens(model, collector_class=LensCollector)  # result.csv, index.csv, config.json - no result.html
```

Probing has no page: its default collector already writes the data files alone.

## Models

Operations accept a `TransformerBridge`, a `HookedTransformer` (on transformer-lens 2 and 3), or either
one wrapped in a `CachedModel`, which keeps activations on disk between runs. See
[Models and caching](models.md).
