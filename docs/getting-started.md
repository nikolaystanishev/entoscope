# Getting started

## Install

entoscope needs Python 3.10 to 3.14 and is published on PyPI:

```bash
pip install entoscope
# or, in a uv project
uv add entoscope
```

To work on entoscope itself, see [Development](development.md).

entoscope depends on PyTorch and on transformer-lens 3 or later, which provides `TransformerBridge`, the
wrapper these docs load models with. Intel macOS is the exception: PyTorch stopped publishing x86_64
macOS wheels after 2.2, which pins that platform to PyTorch 2.2, transformer-lens 2.15, `numpy<2`,
`transformers<5` and Python 3.12 or older, and the installer picks those versions there by itself.
transformer-lens 2.15 has no `TransformerBridge`, so load models as `HookedTransformer` there (see
[Models and caching](concepts/models.md)).

## Load a model

Operations take a `transformer_lens` model. A `TransformerBridge` wraps a HuggingFace model as it is,
so what entoscope reads is HuggingFace's own computation:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.eval()
```

## Make a first run

A logit lens reads the residual stream after every layer, at every position of a prompt, as if the
model stopped there and predicted the next token:

```python
from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig

config = LensConfig(
    method=LensMethodConfig(type="logit_lens"),
    top_k=3,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
    ),
)
lens_input = LensInput(config, "The capital of France is")

Lens(model).apply(lens_input, save_dir="runs/first_lens")
```

The parts of that run:

- `LensConfig` says what to run: the lens method, how many tokens to keep per row, and `source`,
  where in the model to read.
- `source` is a locator: the `resid.post` activation at every `layer` and every `position`. That
  makes 12 × 6 rows for GPT-2 and this prompt (the tokenizer prepends a BOS token, so the prompt has
  6 positions). [Locating activations](concepts/locating.md) covers the locator in full.
- `LensInput` pairs the config with the prompt it reads.
- `Lens(model).apply(...)` runs the model once, reads every row out onto the vocabulary and writes
  the results into `save_dir`.

## Read the results

The run leaves four files behind:

```text
runs/first_lens/
├── result.html   an interactive page: layer up, position across, each cell's top token
├── result.csv    one row per (layer, position), with its top-3 tokens and their probabilities
├── index.csv     the rows the run swept
└── config.json   the config the run was made under
```

Open `result.html` in a browser; it needs no server and no network. `result.csv` is the same result
as a table: the row's location (`source_layer`, `source_position`, `source_activation`), then
`token_i`, `str_token_i` and `probability_i` for each rank `i`:

```python
import pandas as pd

result = pd.read_csv("runs/first_lens/result.csv", keep_default_na=False)
print(result[result.source_position == 5][["source_layer", "str_token_0", "probability_0"]])
```

`keep_default_na=False` keeps tokens such as `"null"` or `"NA"` from being read as missing values.

To keep the results in memory instead, pass `return_collector=True`. The collector's `result` is the
frame `result.csv` holds, and `save_dir` can be left out to write nothing at all:

```python
collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
print(collector.result.shape)
```

## Plan before you run

Every operator can describe a run without making it. `plan` resolves the same rows `apply` would
iterate and reports the work, the activations it would read and the files it would write, without a
forward pass:

```python
plan = Lens(model).plan(lens_input, save_dir="runs/first_lens")
```

```text
Lens plan
  method: logit_lens
  prompt: "The capital of France is" (6 tokens)
  reads: blocks.0.hook_resid_post, blocks.1.hook_resid_post, blocks.10.hook_resid_post, blocks.11.hook_resid_post, blocks.2.hook_resid_post, blocks.3.hook_resid_post, blocks.4.hook_resid_post, blocks.5.hook_resid_post, blocks.6.hook_resid_post, blocks.7.hook_resid_post, +2 more
  work: 72 elements, 1 forward pass
  writes: runs/first_lens/{result.html, result.csv, index.csv, config.json}
  index (72 rows):
     source_layer  source_position source_activation
                0                0        resid.post
                0                1        resid.post
                0                2        resid.post
  (the remaining 69 rows of the index are printed too)
```

The plan is logged as it is built and returned as an `OperationPlan` you can read in code. See
[Dry runs](concepts/running.md#dry-runs).

## Next steps

- [How entoscope is put together](concepts/overview.md): the four parts every operation shares.
- [Locating activations](concepts/locating.md): reading other activations, heads, neurons and
  positions.
- [Configuration files](concepts/configuration.md): the same run from a YAML file.
- The operation pages: [Lens](operations/lens.md), [Probing](operations/probing.md),
  [Activation patching](operations/patching.md), [Circuit discovery](operations/circuit-discovery.md).
- [Examples](examples.md): a runnable script and configs for every operation.
