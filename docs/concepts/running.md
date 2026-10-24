# Running and reading results

## Running

Every operator has the same two calls:

- `apply(input, save_dir=None, return_collector=False, verbose=True)` makes the run.
- `plan(input, save_dir=None, verbose=True)` describes it without making it (see [Dry runs](#dry-runs)).

`apply` arguments:

- `save_dir` is where the files go; it is created if missing. Left as `None`, the run writes nothing.
- `return_collector=True` returns the collector. Its `result` holds what the run collected: a
  DataFrame for the lens, patching and probing, a list of `PromptCircuit` for circuit discovery. A
  collector that draws a [result page](#result-pages) also hands back its `figures`, the page's plotly
  figures keyed by panel (a dict per prompt for circuit discovery), drawn whether or not `save_dir`
  is set.
- `verbose=False` hides the progress bar. Where the results were saved is logged either way (see
  [Logging](#logging)).

## What a run writes

| Operation | Files in `save_dir` |
|---|---|
| Lens | `result.csv`, `index.csv`, `config.json`, `result.html` |
| Activation patching | `result.csv`, `index-0.csv` (one `index-{i}.csv` per patch element), `config.json`, `result.html` |
| Probing | `result.csv`, `index.csv`, `config.json`, `probes.pkl` |
| Circuit discovery | `graph_{i}.graphml`, `graph_{i}.json`, `routes_{i}.graphml`, `routes_{i}.json` and `circuit_{i}.html` per prompt, `prompts.json`, `config.json` |

`result.csv` has one row per row of the index: the index's own columns (see
[Locating activations](locating.md#from-index-to-result)), then the operation's. `index.csv` is the
index alone, and `config.json` the config the run was made under, every field written out with its
defaults, which `from_dict` (in `entoscope.util.data`) reads back into the same config. The operation
pages describe the rest.

### Reading results back

Token strings (the lens's `str_token_i`, patching's `str_generation`) are written with non-printable
characters escaped, so a newline token reads `\n`. Read them with `keep_default_na=False`: otherwise
pandas turns real tokens such as `"null"`, `"None"` or `"NA"` into missing values. The token ids beside
them are exact either way.

<!-- docs-test: skip -->
```python
import pandas as pd

result = pd.read_csv("runs/first_lens/result.csv", keep_default_na=False)
```

From a shell, `less -S` pages through a result without wrapping its long rows, and `head` and `tail`
print its first and last rows:

```bash
less -S runs/first_lens/result.csv
head -n 5 runs/first_lens/result.csv
tail -n 5 runs/first_lens/result.csv
```

## Result pages

The lens, patching and circuit discovery draw their results as a self-contained interactive page with
[Plotly](https://plotly.com/python/). The page carries plotly.js inside it, so it opens straight from
the file, with no server and no network.

A lens or patching page draws the run's index as a grid:

- When both `layer` and `position` vary, position goes across and layer up, the usual
  activation-patching layout.
- Otherwise the varying axes are taken in the order `layer`, `head`, `neuron`, `position`,
  `dest_pos`, `src_pos`, and the last of them goes across. A head's attention pattern is thus drawn
  the way patterns usually are, `src_pos` across and `dest_pos` up.
- An axis the run does not vary over is not drawn. More than two varying axes fold into composite
  rows, such as `layer 5 · head 3`.
- An index that varies along one axis, including one built from explicit `coordinates`, becomes a bar
  chart.
- Position ticks, `dest_pos` and `src_pos` ones included, carry the prompt's tokens, such as
  `4: " Spain"`. For patching that is the target prompt, since the patched run is the target run.
- A cell's hover gives its full coordinates.

A page holds one plot per panel: a lens draws one per top-k rank and patching one per result column.
Circuit discovery writes a page per prompt, with the routes to the prediction and then the full
contribution graph. The operation pages describe what each plot encodes.

The page is a collector's choice rather than a config flag. Each operator's default collector draws
it, and the base collector writes the data files alone:

<!-- docs-test: skip -->
```python
from entoscope.circuit import CircuitCollector
from entoscope.lens import LensCollector
from entoscope.patching import PatchingCollector

Lens(model, collector_class=LensCollector)
Patcher(model, collector_class=PatchingCollector)
CircuitDiscoverer(model, collector_class=CircuitCollector)
```

## Dry runs

`plan` resolves the input's index, the same rows `apply` would iterate, and reports the work, the
activations it would read and the files it would write, without a forward pass and without touching
`save_dir`:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import ConfigFactory, Lens, LensConfig, LensInput

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

lens_input = LensInput(
    ConfigFactory[LensConfig]("examples/lens.yaml").create("logit_lens"), "The capital of France is"
)
plan = Lens(model).plan(lens_input, save_dir="runs/logit_lens", verbose=False)
print(plan.steps, plan.forward_passes, plan.outputs)  # 72 1 ['result.html', 'result.csv', 'index.csv', 'config.json']
```

With `verbose=True`, the default, the plan is also logged as text (as in
[Getting started](../getting-started.md#plan-before-you-run)). The returned `OperationPlan` holds the
same facts: `steps`, `forward_passes`, `backward_passes`, `batches`, `prompts`, `activations`,
`outputs`, and `index`, one DataFrame per index holding exactly what `index.csv` would.
`plan.describe(max_rows=...)` renders it with more or fewer index rows.

Passing the same input object to `apply` afterwards does not resolve the index again: only the
activations are still to be gathered.

What a plan counts, per operator:

| Operator | A step is | Forward passes | Backward passes |
|---|---|---|---|
| `Lens` | an index row | 1, or 0 for a lens that reads weights | 0 |
| `Patcher` with `exact` | a patched run, one per index row | one per batch, plus one per prompt | 0 |
| `Patcher` with `attribution` | an index row | 1, plus one per prompt | one per metric |
| `Patcher` with `integrated_gradients` | an index row | `(interpolation_steps - 1) × activations + 1`, plus one per prompt | `(interpolation_steps - 1) × activations + 1` per metric |
| `Prober` | a probe, one per index row | one per example prompt | 0 |
| `CircuitDiscoverer` | a prompt | one per prompt | 0 |

`activations` is the number of transformer activations the patch writes, such as one per layer of a
sweep.

In an experiment script, a `dry_run` field on your own config gives the script a `--dry-run` flag; see
[Experiment scripts](../guides/experiment-scripts.md).

## Logging

What entoscope reports, such as a plan or where a run's results were saved, goes through Python's
`logging`, under the `entoscope` logger at INFO, to stderr. A script's stdout stays its own:

```text
[entoscope] INFO 2026-10-10 14:03:13 Results saved to runs/logit_lens
```

The `entoscope` logger has its own handler and does not propagate to the root logger. Its messages
therefore keep this form whatever configures root logging, including dependencies: transformer_lens
logs through the module-level `logging.info`, which runs `basicConfig` as a side effect. To silence
entoscope:

<!-- docs-test: skip -->
```python
import logging

logging.getLogger("entoscope").setLevel(logging.WARNING)
```

To route its messages through your application's handlers instead:

<!-- docs-test: skip -->
```python
entoscope_logger = logging.getLogger("entoscope")
entoscope_logger.handlers.clear()
entoscope_logger.propagate = True
```
