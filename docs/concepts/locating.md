# Locating activations

Lens, patching and probing read activations at locations a **locator** names: the `source` of a
lens config, the `source` and `target` of every patch element, and the `source` of a probe config. A
locator is a `ModelLocatorConfig`. Resolving it against the model and the prompt gives the run's **index**: one
row per location, which is also one row of the result.

The examples on this page build locators in code, resolve them against GPT-2 and print the rows they
give. A config file writes the same locators with the enums' values, as the YAML blocks show:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import ModelAxis, ModelLayerLocation, ModelLocatorConfig
from entoscope.model.index import Index

model = TransformerBridge.boot_transformers("gpt2")
model.eval()


def rows(locator: ModelLocatorConfig, n_positions: int = 6):
    """The rows a locator resolves to for a prompt of n_positions tokens, as index.csv holds them."""
    return Index(locator, locator.parse(model, n_positions)).to_df()


locator = ModelLocatorConfig(
    activation=ModelLayerLocation.RESIDUAL_POST,
    location=[ModelAxis.LAYER, ModelAxis.POSITION],
)
print(rows(locator))  # 72 rows: 12 layers x 6 positions
```

## Activations

`activation` names the activation a row reads, a [`ModelLayerLocation`](#entoscope.model.config.ModelLayerLocation) member. Each one is a
`transformer_lens` hook:

| `activation` | `ModelLayerLocation` | Axes of the tensor | Hook (in layer 0) | Size of `neuron` |
|---|---|---|---|---|
| `embed` | `EMBEDDING` | batch, position, neuron | `hook_embed` | `d_model` |
| `resid.pre` | `RESIDUAL_PRE` | batch, position, neuron | `blocks.0.hook_resid_pre` | `d_model` |
| `resid.mid` | `RESIDUAL_MID` | batch, position, neuron | `blocks.0.hook_resid_mid` | `d_model` |
| `resid.post` | `RESIDUAL_POST` | batch, position, neuron | `blocks.0.hook_resid_post` | `d_model` |
| `attn.q` | `ATTENTION_Q` | batch, position, head, neuron | `blocks.0.attn.hook_q` | `d_head` |
| `attn.k` | `ATTENTION_K` | batch, position, head, neuron | `blocks.0.attn.hook_k` | `d_head` |
| `attn.v` | `ATTENTION_V` | batch, position, head, neuron | `blocks.0.attn.hook_v` | `d_head` |
| `attn.pattern` | `ATTENTION_PATTERN` | batch, head, query position, key position | `blocks.0.attn.hook_pattern` | none |
| `attn.z` | `ATTENTION_Z` | batch, position, head, neuron | `blocks.0.attn.hook_z` | `d_head` |
| `attn.out` | `ATTENTION_OUT` | batch, position, neuron | `blocks.0.hook_attn_out` | `d_model` |
| `mlp.pre` | `MLP_PRE` | batch, position, neuron | `blocks.0.mlp.hook_pre` | `d_mlp` |
| `mlp.pre_linear` | `MLP_PRE_LINEAR` | batch, position, neuron | `blocks.0.mlp.hook_pre_linear` | `d_mlp` |
| `mlp.post` | `MLP_POST` | batch, position, neuron | `blocks.0.mlp.hook_post` | `d_mlp` |
| `mlp.out` | `MLP_OUT` | batch, position, neuron | `blocks.0.hook_mlp_out` | `d_model` |

A few of them have quirks:

- `embed` sits before the first block, so its locator names no `layer`, and one that does is refused.
- `attn.pattern` is located by `layer` and `head` only. Its two position axes are always read whole,
  and it has no `neuron` axis, so a locator naming `position` or `neuron` for it is refused rather
  than reading the same whole pattern on every row.
- `attn.k` and `attn.v` have `n_key_value_heads` heads on a model with grouped-query attention.
- `mlp.pre_linear` exists only on a gated MLP, where it is the linear branch and `mlp.pre` the gate.

## Axes

A row names a value on some of four axes, the members of [`ModelAxis`](#entoscope.model.config.ModelAxis):

- `layer` (`ModelAxis.LAYER`): `0` to `n_layers - 1`.
- `head` (`ModelAxis.HEAD`): `0` to `n_heads - 1`.
- `neuron` (`ModelAxis.NEURON`): the activation's last axis, sized as in the table above.
- `position` (`ModelAxis.POSITION`): `0` to the prompt's token count minus one. `model.to_tokens` prepends a BOS token, so
  position 0 is BOS and a prompt's first word is at position 1.

A negative value counts from the end of its axis: `position: -1` is the last token and `layer: -1` the
last layer. A value past either end of its axis is refused when the index is resolved, by `plan()` as
by a run, so `layer: 12` on GPT-2's twelve layers fails before any forward pass. An axis a row does not
name is read whole: every row of `{activation: resid.post, location: [layer]}` spans every position.

A locator that names any axis needs an `activation` to read it in, set on the locator or on every
coordinate, and may name only the axes of that activation's tensor, plus `layer` for every activation
but `embed`: `head` on `resid.post`, say, is refused as the config is read.

## Two ways to name locations

A locator names its rows in one of two ways, never both:

- **Declaratively**: `location` lists axes, every value of each is enumerated, and `filter`, `group`,
  `incremental` and `combination` narrow and reshape the grid.
- **Explicitly**: `coordinates` lists the exact rows.

### Declarative: axes, then a filter

`location` lists the axes the index enumerates, and the index is every combination of their values:
`[layer, head]` on `attn.z` is 12 × 12 rows for GPT-2. A `filter` then keeps the rows matching any of
its terms. Within a term every key must match, and a term `{not: ...}` drops the rows it matches
instead. A constraint is an int, `{range: [start, end]}` with both ends included, or a list of these:

```python
from entoscope import Filter

locator = ModelLocatorConfig(
    activation=ModelLayerLocation.ATTENTION_Z,
    location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.POSITION],
    filter=Filter.from_config(
        [
            {ModelAxis.LAYER: {"range": [0, 5]}, ModelAxis.POSITION: -1},  # layers 0-5, last position only
            {ModelAxis.LAYER: 11},  # or layer 11, every position
            {"not": {ModelAxis.HEAD: 0}},  # minus every row of head 0
        ]
    ),
)
print(len(rows(locator)))  # 132: (6 layers x 1 position + 1 layer x 6 positions) x 11 heads
```

The same locator in a config file:

```yaml
source:
  activation: attn.z
  location: [layer, head, position]
  filter:
    - layer: {range: [0, 5]}
      position: -1
    - layer: 11
    - not: {head: 0}
```

### Reading several values together: group, incremental, combination

A row can hold a list of values on an axis. It then reads all of them together. A lens sums a group of
heads or neurons into one direction, a patch writes the whole group in one run, and a probe takes
every one of them as features. Three transforms build such rows, applied after the filter in this
order:

- `group` merges the listed values of an axis into one row; `all` merges every value. The axis has to
  be one `location` enumerates.
- `incremental` follows every row with rows that add growing prefixes of the listed values, adding the
  axis when `location` does not name it.
- `combination` expands every row into one row per combination of the values it holds on the listed
  axes, from one value up to `max_elements`. Each axis has to be one `location` or `incremental` names.

```python
from dataclasses import replace

from entoscope import Combination, Group, Incremental

heads = ModelLocatorConfig(activation=ModelLayerLocation.ATTENTION_Z, location=[ModelAxis.LAYER, ModelAxis.HEAD])

print(rows(replace(heads, filter=Filter.from_config([{ModelAxis.LAYER: 9}]), group=Group({ModelAxis.HEAD: [6, 9]}))))
# 11 rows: one reading heads 6 and 9 together, and one for each other head of layer 9

print(
    rows(
        replace(
            heads,
            filter=Filter.from_config([{ModelAxis.LAYER: 9, ModelAxis.HEAD: 0}]),
            incremental=Incremental({ModelAxis.HEAD: [6, 9]}),
        )
    )
)
# 3 rows: heads [0], [0, 6] and [0, 6, 9]

print(
    rows(
        replace(
            heads,
            filter=Filter.from_config([{ModelAxis.LAYER: 9, ModelAxis.HEAD: [6, 9, 10]}]),
            group=Group({ModelAxis.HEAD: "all"}),
            combination=Combination(location=[ModelAxis.HEAD], max_elements=2),
        )
    )
)
# 6 rows: (6,), (9,), (10,), (6, 9), (6, 10), (9, 10)
```

Every layer is a hook of its own, so a row reads one layer. Group heads, neurons and positions, but
not layers: a row that names several layers names no activation a run can read.

### Explicit: coordinates

`coordinates` lists the rows themselves, each a mapping from axis to value. Every coordinate uses the
same keys, a value may be a list to read several entries together, and negative values count from
the end. A coordinate may name its own `activation`, and then every coordinate must, while the
locator's `activation` stays unset:

```python
from entoscope import Coordinates

locator = ModelLocatorConfig(
    coordinates=Coordinates(
        [
            {ModelAxis.ACTIVATION: ModelLayerLocation.RESIDUAL_POST, ModelAxis.LAYER: -2, ModelAxis.POSITION: -1},
            {ModelAxis.ACTIVATION: ModelLayerLocation.MLP_OUT, ModelAxis.LAYER: -1, ModelAxis.POSITION: -1},
        ]
    )
)
print(rows(locator))
# 2 rows, in the order listed: resid.post at layer 10 and mlp.out at layer 11, both at position 5
```

Coordinates cannot be combined with `location`, `filter` or the transforms.

## Prompts of different lengths

A lens or a patching run resolves its positions against its prompt. Probing reads many prompts, which
may tokenize to different lengths, and then positions stay symbolic. A non-negative position counts
from the start of every prompt, and a negative one from the end of each prompt in turn. Three things
follow:

- An unfiltered `position` axis is rejected, since its size would depend on the prompt. Filter it to
  explicit positions, or leave `position` out of `location` and pool over it (see
  [Probing](../operations/probing.md)).
- A `range` must stay on one side: `{range: [-3, -1]}` works, `{range: [1, -1]}` does not.
- Every explicit position must exist in the shortest prompt.

## From index to result

The index is written to `index.csv` and makes up the first columns of `result.csv`. Every axis
becomes a column prefixed `source_` (and `target_` for the target side of a patch), with the
activation in `source_activation`. A row's list of values is written as a list, such as `[6, 9]` or
`(6, 9)`. The operation's own columns follow; each operation page lists them.
