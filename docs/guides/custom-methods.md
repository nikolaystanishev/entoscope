# Custom methods

Every operation's methods are classes in a registry, and the config's `method.type` picks one by
name. The built-in methods are registered the same way you register yours, so a method you write runs
through the same operator, plan, collector and result page as theirs.

## A method in three steps

This lens divides the logit lens's logits by a temperature before the collector's softmax:

```python
from dataclasses import dataclass

from transformer_lens.model_bridge import TransformerBridge

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig


# 1. The method's settings, as a dataclass extending the operation's method config.
@dataclass(kw_only=True)
class TemperatureLensConfig(LensMethodConfig):
    temperature: float = 1.0


# 2. The method, naming its settings as its type argument. 3. Registered under a name.
@LensMethodFactory.register("temperature_lens")
class TemperatureLens(LogitLens[TemperatureLensConfig]):
    def project(self, activation, index):
        return super().project(activation, index) / self.method_config.temperature


model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = LensConfig(
    method=TemperatureLensConfig(type="temperature_lens", temperature=2.0),
    top_k=5,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER, ModelLocation.POSITION]
    ),
)
Lens(model).apply(LensInput(config, "The capital of France is"), save_dir="runs/warm")
```

A config file names it like a built-in method:

```yaml
warm:
  method:
    type: temperature_lens
    temperature: 2.0
  top_k: 5
  source:
    activation: resid.post
    location: [layer, position]

plain:
  method: temperature_lens  # just the type: the method with its defaults
  source:
    activation: resid.post
    location: [layer, position]
```

[`examples/custom_lens.py`](https://github.com/nikolaystanishev/entoscope/blob/main/examples/custom_lens.py)
is this lens as an experiment script, run from a config file.

## What each operation asks of a method

| Operation | Register with | Extend | Settings extend | Implement | Built |
|---|---|---|---|---|---|
| Lens | `LensMethodFactory` | `AbstractLensMethod` | `LensMethodConfig` | `extract`, `project` | once per run |
| Probing | `ProbeMethodFactory` | `AbstractProbeMethod` | `ProbeMethodConfig` | `fit`, `predict`, `predict_proba`, `probe` | once per index row |
| Activation patching | `PatchingMethodFactory` | `AbstractPatchingMethod` | `PatchingMethodConfig` | `scores`, `passes` | once per run |
| Circuit discovery | `CircuitMethodFactory` | `AbstractCircuitMethod` | `CircuitMethodConfig` | `build` | once per run |

The factories live in `entoscope.<operation>.method.factory`, the method bases in
`entoscope.lens.method.lens`, `entoscope.probe.method.probe`, `entoscope.patching.method.patching` and
`entoscope.circuit.method.circuit`, and the settings bases in `entoscope.<operation>.config`. A method is
built from `(model, config)` (patching adds `batch_size`) and finds its own settings in
`self.method_config`.

- **Lens.** One instance reads every row of the run, so a lens that handles several kinds of
  activation tells them apart with `index.activation()`. `extract(lens_input, index)` returns the
  tensor a row names; the logit lens's is `lens_input.cache[index.transformer_activation()][index.slices()]`.
  `project(tensor, index)` maps it to the row's logits, a single vector over the vocabulary. Extending
  `LogitLens`, as above, keeps its read-out of residual-stream, MLP and attention head activations.
  A lens that reads only the weights sets `reads_activations = False`: it then takes no prompt and
  has no `position` axis, like `ParameterLens`.
- **Probing.** `fit(features, labels)` trains on a `[examples, features]` tensor and 0/1 labels.
  `predict` returns the 0/1 predictions and `predict_proba` the positive-class probabilities, both as
  numpy arrays. `probe` is the fitted estimator the collector pickles into `probes.pkl`, so it must
  pickle.
- **Activation patching.** `scores(patching_input, verbose)` yields a `RowResult` per patched run
  (`patching_input.rows()` lists them). Each one is built from the change the run made to every metric,
  the patched run's value minus the unpatched target run's, which `self._report` turns into the columns
  the config's comparisons ask for. `passes(runs, batches)` returns the `(forward, backward)` passes
  the method adds, for `plan`. Set `batched = True` if it runs the model `batch_size` rows at a time.
  The test suite's
  [`test_custom_patching_method.py`](https://github.com/nikolaystanishev/entoscope/blob/main/tests/patching/test_custom_patching_method.py)
  is a minimal worked example.
- **Circuit discovery.** `build(prompt, tokens)` returns the prompt's `PromptCircuit`. The result page
  reads the node and edge attributes Information Flow Routes writes (see
  [The graph](../operations/circuit-discovery.md#the-graph)); a method writing other attributes runs
  with `collector_class=CircuitCollector`.

## Settings

A method's type argument names the dataclass of its settings. That one name types
`self.method_config` and is the class a config file's `method` is read into.

- A method with no settings leaves the type argument out, and takes its operation's base, such as
  `LensMethodConfig`, which no field fits.
- A method extending another keeps that one's settings. The exception is a method extending one that
  has none of its own (`logit_lens`, `exact`, `attribution`): those leave their type argument open, so
  the subclass names its own, as `TemperatureLens` does.
- Settings are data. They are written to `config.json` and read back from it, so they hold numbers,
  strings, enums and paths, and lists, tuples, sets, mappings and dataclasses of them. A method that
  needs something else, such as a function or a tensor, builds it from its settings (a path to load it
  from, say) or holds it as a class attribute.
- A setting a config file may leave out needs a default. In code, build the instance of exactly the
  method's settings class, since `config.json` is read back into that class.
- `type` is declared once, on the shared base, with no default: one settings class can serve several
  methods, so the instance always names its method.

## Registration

A method is registered when the module defining it is imported, so import it before you build or load
a config that names it. A config file looks its method up as it is read: a name nothing registered
fails right there, listing the names that are registered, before any run. The built-in methods
register when their operation's package is imported, so they are always known.

- In an [experiment script](experiment-scripts.md), define the method above `main`, since `main` runs
  as it is decorated.
- A name belongs to one method. Registering another class under a taken name raises, so a built-in
  cannot be shadowed by accident.
- Defining the same class again, by re-running a notebook cell or reloading its module, replaces it.
  A class built inside a function is a new class at every call.
- A class that leaves one of its base's abstract methods out is rejected when it is registered.
- One class may be registered under several names.

## Collectors and figures

`collector_class` is an extension point too. A collector subclasses `FrameCollector` (one result row
per index row), implements `collect` and `_indices`, and declares the files it writes in
`output_files` so that `plan` can report them. To draw a page as well, a collector extends
`VisualizationCollector` and names a `GridFigure` subclass, which says what a panel plots: its
`value_column`, its `hover` text, its title, and optionally its colour range. Both live in
`entoscope.operation`; see the [API reference](../reference/operation.md).
