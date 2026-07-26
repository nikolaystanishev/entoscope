# intex

A mechanistic-interpretability toolkit for large language models, built on
[`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens). Each operation follows
the same shape — `Config → Input → Operator → Collector` — and can be driven inline or from an
experiment script. Current operations: Logit Lens, Tuned Lens, Parameter Lens, Probing, Activation
Patching, and Circuit Discovery.

## Models

Operations take a `TransformerBridge`, the wrapper `transformer_lens` loads HuggingFace models
through, and read it as-is — so activations and weights are HuggingFace's own, and a run means the
same thing as the same computation done against the HuggingFace model directly:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.eval()
```

Calling `model.enable_compatibility_mode()` switches the model into `transformer_lens`'s convention
instead, folding LayerNorm into the following weights and centering the ones that write to the
residual stream. Both are supported; the choice changes what the intermediate numbers mean, not
whether a run works. Two consequences worth knowing:

- Under HuggingFace's numerics a model that ties its embedding and unembedding — GPT-2 does — has
  `W_E.T == W_U`, so the parameter lens's two `projection` spaces coincide. Folding separates them.
- `eval()` on a bridge does not reach the HuggingFace model it wraps. `boot_transformers` returns
  one already in eval mode, but a model you construct yourself and pass as `hf_model=` needs its
  own `eval()` or dropout stays on.

`HookedTransformer` is still accepted wherever a bridge is — `Model` is the union of the two, plus
the `CachedModel` wrapper — and the test suite runs against both. It is the only option on
platforms whose `transformer_lens` predates the bridge.

## Parameter Lens

Reads the model's **weights** in vocabulary space, following
[Analyzing Transformers in Embedding Space](https://arxiv.org/abs/2209.02535): a `d_model`
parameter vector is projected onto the vocabulary directly, without LayerNorm or biases. Nothing
about it depends on an input, so a run takes no prompt at all — it is the same `Lens` operator and
the same `LensInput`, with `prompt=None`.

```python
from transformer_lens.model_bridge import TransformerBridge

from intex.lens.config import LensConfig, LensType
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = LensConfig(
    type=LensType.PARAMETER_LENS,
    top_k=5,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER, ModelLocation.NEURON],
        filter=Filter(inclusion_terms=[{ModelLocation.NEURON: 373}]),
    ),
)

collector = Lens(model).apply(
    LensInput(config), save_dir="runs/parameter_lens", return_collector=True
)
print(collector.result)  # one row per (layer, neuron), with its top-k tokens
```

Alongside `result.csv` the run writes an interactive `result.html` — see
[Result visualization](#result-visualization).

An index element needs a single layer — each weight lives in one block — while a list of neurons
(or heads) sums those weight vectors into one direction before projecting it, and leaving the
location out sums all of them, exactly as grouped neurons and heads work in the logit lens.
`position` is not a valid location:

| location                       | weight                                     | reading                              |
|--------------------------------|--------------------------------------------|--------------------------------------|
| `mlp.pre`                      | column of `W_in` (`W_gate` on a gated MLP) | the key vector the neuron reads with |
| `mlp.pre_linear`               | column of `W_in` (gated MLPs only)         | the key vector of the linear branch  |
| `mlp.post`                     | row of `W_out`                             | the value vector the neuron writes   |
| `attn.q` / `attn.k` / `attn.v` | column of `W_Q` / `W_K` / `W_V`            | what the subhead reads               |
| `attn.z`                       | row of `W_O`                               | what the subhead writes              |

`projection` picks the vocabulary space: `unembedding` (the default) multiplies by `W_U`,
`embedding` by the embedding matrix `E`, as in the paper. The two differ whenever a model does not
tie them — GPT-2 does tie them, so on HuggingFace's weights the two spaces agree exactly and only
come apart once `enable_compatibility_mode` has folded LayerNorm in and centered the unembedding.
The collector's `probability_*` columns are a softmax over the projected weight
vector: a ranking score over the vocabulary, not a probability the model ever produces.

`scale` multiplies the weight vector before projecting it (default `1.0`). A negative scale such as
`-1.0` inverts the direction, surfacing the tokens a weight writes *against* rather than *toward*;
the magnitude acts as a temperature on the `probability_*` ranking. It applies only to the
parameter lens.

Configs can also be loaded from YAML via `intex.operation.config.ConfigFactory[LensConfig]` (see
`example_config.yaml`), and the operation can be run from an `@intex.script.experiment.experiment`
script (see `example_lens_script.py`).

## Circuit Discovery

Discovers the circuit a model uses on a set of prompts. It supports multiple methods; the first is
**Information Flow Routes**, which builds a contribution graph over the residual stream and prunes the
routes that lead to the model's prediction.

```python
from transformer_lens.model_bridge import TransformerBridge

from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.circuit.discoverer import CircuitDiscoverer
from intex.circuit.input import CircuitInput

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = CircuitConfig(type=CircuitMethodType.INFORMATION_FLOW_ROUTES, threshold=0.04)
prompts = ["The capital of France is", "The Eiffel Tower is in"]

collector = CircuitDiscoverer(model).apply(
    CircuitInput(config, prompts), save_dir="runs/circuit", return_collector=True
)

# In memory: one PromptCircuit per prompt, each with a full graph and the pruned routes.
first = collector.result[0]
print(first.graph, first.routes)
```

For each prompt `i`, the collector writes into `save_dir`:

- `graph_{i}.graphml` / `graph_{i}.json` — the full thresholded contribution graph (a
  `networkx.MultiDiGraph` with a separate attention edge per head).
- `routes_{i}.graphml` / `routes_{i}.json` — the pruned subgraph of routes to the prediction.
- `circuit_{i}.html` — a self-contained interactive Plotly visualization (tokens on the x axis,
  layers on the y axis; attention edges coloured by head, sized by contribution).
- `prompts.json`, `config.json` — the run's index and configuration.

By default `CircuitDiscoverer` uses `CircuitVisualizationCollector`, which writes the HTML above. For
graph-data-only runs (no HTML), pass the base collector:
`CircuitDiscoverer(model, collector_class=CircuitCollector)` — see
[Result visualization](#result-visualization), which every operation shares.

Configs can also be loaded from YAML via `intex.operation.config.ConfigFactory[CircuitConfig]`, and
the operation can be run from an `@intex.script.experiment.experiment` script.

## Result visualization

Every operation writes a self-contained interactive Plotly page into `save_dir` next to its data —
`result.html` for a lens or patching run, `circuit_{i}.html` per prompt for circuit discovery. The
page needs no network and no local server: open the file.

A lens or patching page draws the run's own index. **Position goes on the x axis and layer on the y
axis** when both are present — the canonical activation-patching layout — otherwise the axes follow
`layer, head, neuron, position`, with the last of these the run varies over going across. An axis the
run does not vary over is dropped, further axes fold into composite y ticks (`layer 5 · head 3`), and
a one-dimensional index — including one built from explicit `coordinates` — becomes a bar chart.
Position ticks carry the prompt's tokens (`4: " Spain"`); for patching that is the *target* prompt,
since the patched run is the target run.

What each encodes:

- **Patching** — the metric on a diverging scale centred on the value an ineffective patch would
  produce (the unpatched target's own logit diff, or zero for the normalized and KL metrics), so warm
  cells are the locations that moved the model toward the source. Hover carries the metric, the full
  source and target coordinates, and what any other patch element did.
- **Lens** — the top-1 probability on a sequential scale fixed to `[0, 1]`, with the top-1 token
  printed in the cell and the whole top-k in hover.

Visualization is chosen by the collector rather than by a config flag, so a run that only needs the
CSV opts out by naming the base collector — and only then does the operation avoid importing plotly:

```python
Patcher(model, collector_class=PatchingCollector)
Lens(model, collector_class=LensCollector)
CircuitDiscoverer(model, collector_class=CircuitCollector)
```

## Dry run

Every operator can describe a run before making it. `plan` resolves the input's index — the same
expansion `apply` iterates — and reports the work, the activations it would read and the files it
would write, without a single forward pass and without touching `save_dir`:

```python
lens_input = LensInput(config, "The capital of France is")

plan = Lens(model).plan(lens_input, save_dir="runs/logit_lens")
```

```text
Lens plan — logit_lens (config: my_lens)
  prompt: "The capital of France is" (6 tokens)
  reads: blocks.0.hook_resid_post, blocks.1.hook_resid_post, blocks.10.hook_resid_post, ...
  work: 72 elements, 1 forward pass
  writes: runs/logit_lens/{result.html, result.csv, index.csv, config.json}
  index (72 rows):
     source_layer  source_position source_activation  index_layer  index_position
                0                0        resid.post            0               0
                0                1        resid.post            0               1
              ...              ...               ...          ...             ...
```

What each operator reports:

| operator            | steps                                  | forward passes                                              |
|---------------------|----------------------------------------|-------------------------------------------------------------|
| `Lens`              | one element per index row              | 1 — none at all for the parameter lens, which reads weights |
| `Prober`            | one probe per index row                | one per example prompt                                      |
| `Patcher`           | one patched run per source/target pair | one per batch, plus the two cached runs                     |
| `CircuitDiscoverer` | one graph per prompt                   | one per prompt                                              |

The plan is printed as it is built; pass `verbose=False` to keep it quiet and only take the object
back. That object, an `OperationPlan`, is a dataclass, so the same numbers are readable in code:
`steps`, `forward_passes`, `batches`, `activations`, `outputs`, and `index` — one DataFrame per
index, holding exactly what `index.csv` would. `plan.describe(max_rows=...)` re-renders it with
more or less of the index than the printout shows.

Passing the *same* input object to `apply` afterwards costs nothing extra: the resolved index is
reused, and only the activations are still fetched.

In an experiment script, put the switch on your own config and `--dry-run` comes for free from the
parser `@experiment` builds (see `example_lens_script.py`):

```python
@dataclass
class Config:
    model_name: str
    dry_run: bool = False


@experiment
def main(config: Config):
    model = TransformerBridge.boot_transformers(config.model_name)
    lens_input = LensInput(config_of_the_run, "The capital of France is")

    if config.dry_run:
        Lens(model).plan(lens_input, "runs/logit_lens")
    else:
        Lens(model).apply(lens_input, save_dir="runs/logit_lens")
```

## Datasets

An extension point for declaring *where* prompts come from, following the same `Config → Factory`
shape as the operations. A dataset yields plain `dict` rows; `intex` ships no concrete
implementation — you register your own and the factory resolves it by the config's `type`:

```python
from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory
from intex.operation.config import ConfigFactory


@DatasetFactory.register("jsonl")
class JsonlDataset(AbstractDataset):
    def load(self) -> list[Row]:
        with open(self.config.path) as f:  # 'path' is a dynamic config property
            return [json.loads(line) for line in f][: self.config.get("limit")]


config = ConfigFactory[DatasetConfig]("example_config.yaml").create("my_data")
dataset = DatasetFactory.create(config).prepare()
print(len(dataset), dataset[0])
```

```yaml
my_data:
  type: jsonl
  path: data/prompts.jsonl
  limit: 10
```

`DatasetConfig` declares only `type`; every other key in the entry is carried as a dynamic property,
readable as `config.path` or `config.get("limit", 100)` — so a dataset defines its own vocabulary
without `intex` knowing about it. Subclass `DatasetConfig` with real dataclass fields when you want
those settings typed and validated instead; unknown keys still land in `properties`.

`load` is called once, lazily, by `prepare()`; `len`, iteration and indexing read the cached rows.
Operations do not consume datasets yet — `LensInput`, `ProbeInput`, `CircuitInput` and
`PatchingInput` still take prompts directly.

## Attribution & licensing

The Information Flow Routes method reuses contribution-graph math from the
[LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool)
(`src/intex/circuit/vendor/contributions.py`, vendored — see `src/intex/circuit/vendor/NOTICE`),
which is licensed **CC BY-NC 4.0**. Because that code is non-commercial, the Circuit Discovery
`information_flow_routes` method — and any results produced with it — may be used for
**non-commercial purposes only**. Method reference:

> Javier Ferrando and Elena Voita. "Information Flow Routes: Automatically Interpreting Language
> Models at Scale." 2024. https://arxiv.org/abs/2403.00824
