# Locating activations

Lens, probing and patching read activations at locations a **locator** names: the `source` of a
lens or probe config, and the `source` and `target` of every patch element. A locator is a
`ModelLocatorConfig`. Resolving it against the model and the prompt gives the run's **index**: one
row per location, which is also one row of the result.

The examples on this page resolve locators against GPT-2 and print the rows they give. They write
each locator as a config file does, a mapping that `from_dict` reads into a `ModelLocatorConfig`:

```python
from transformer_lens.model_bridge import TransformerBridge

from intex.model.config import ModelLocatorConfig
from intex.model.index import Index
from intex.util.data import from_dict

model = TransformerBridge.boot_transformers("gpt2")
model.eval()


def rows(locator: dict, n_positions: int = 6):
    """The rows a locator resolves to for a prompt of n_positions tokens, as index.csv holds them."""
    config = from_dict(ModelLocatorConfig, locator)
    return Index(config, config.parse(model, n_positions)).to_df()


print(rows({"activation": "resid.post", "location": ["layer", "position"]}))  # 72 rows: 12 layers x 6 positions
```

## Activations

`activation` names the activation a row reads. Each one is a `transformer_lens` hook:

| `activation` | Axes of the tensor | Hook (in layer 0) | Size of `neuron` |
|---|---|---|---|
| `embed` | batch, position, neuron | `hook_embed` | `d_model` |
| `resid.pre` | batch, position, neuron | `blocks.0.hook_resid_pre` | `d_model` |
| `resid.mid` | batch, position, neuron | `blocks.0.hook_resid_mid` | `d_model` |
| `resid.post` | batch, position, neuron | `blocks.0.hook_resid_post` | `d_model` |
| `attn.q` | batch, position, head, neuron | `blocks.0.attn.hook_q` | `d_head` |
| `attn.k` | batch, position, head, neuron | `blocks.0.attn.hook_k` | `d_head` |
| `attn.v` | batch, position, head, neuron | `blocks.0.attn.hook_v` | `d_head` |
| `attn.pattern` | batch, head, query position, key position | `blocks.0.attn.hook_pattern` | none |
| `attn.z` | batch, position, head, neuron | `blocks.0.attn.hook_z` | `d_head` |
| `attn.out` | batch, position, neuron | `blocks.0.hook_attn_out` | `d_model` |
| `mlp.pre` | batch, position, neuron | `blocks.0.mlp.hook_pre` | `d_mlp` |
| `mlp.pre_linear` | batch, position, neuron | `blocks.0.mlp.hook_pre_linear` | `d_mlp` |
| `mlp.post` | batch, position, neuron | `blocks.0.mlp.hook_post` | `d_mlp` |
| `mlp.out` | batch, position, neuron | `blocks.0.hook_mlp_out` | `d_model` |

A few of them have quirks:

- `embed` sits before the first block, so its locator names no `layer`. With a layer it would name a
  hook that does not exist.
- `attn.pattern` is located by `layer` and `head` only. Its two position axes are always read whole,
  and it has no `neuron` axis.
- `attn.k` and `attn.v` have `n_key_value_heads` heads on a model with grouped-query attention.
- `mlp.pre_linear` exists only on a gated MLP, where it is the linear branch and `mlp.pre` the gate.

## Axes

A row names a value on some of four axes:

- `layer`: `0` to `n_layers - 1`.
- `head`: `0` to `n_heads - 1`.
- `neuron`: the activation's last axis, sized as in the table above.
- `position`: `0` to the prompt's token count minus one. `model.to_tokens` prepends a BOS token, so
  position 0 is BOS and a prompt's first word is at position 1.

A negative value counts from the end of its axis: `position: -1` is the last token and `layer: -1` the
last layer. An axis a row does not name is read whole: every row of
`{activation: resid.post, location: [layer]}` spans every position.

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
locator = {
    "activation": "attn.z",
    "location": ["layer", "head", "position"],
    "filter": [
        {"layer": {"range": [0, 5]}, "position": -1},  # layers 0-5, last position only
        {"layer": 11},  # or layer 11, every position
        {"not": {"head": 0}},  # minus every row of head 0
    ],
}
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

- `group` merges the listed values of an axis into one row; `all` merges every value.
- `incremental` follows every row with rows that add growing prefixes of the listed values.
- `combination` expands every row into one row per combination of the values it holds on the listed
  axes, from one value up to `max_elements`.

```python
heads = {"activation": "attn.z", "location": ["layer", "head"]}

print(rows({**heads, "filter": [{"layer": 9}], "group": {"head": [6, 9]}}))
# 11 rows: one reading heads 6 and 9 together, and one for each other head of layer 9

print(rows({**heads, "filter": [{"layer": 9, "head": 0}], "incremental": {"head": [6, 9]}}))
# 3 rows: heads [0], [0, 6] and [0, 6, 9]

print(
    rows(
        {
            **heads,
            "filter": [{"layer": 9, "head": [6, 9, 10]}],
            "group": {"head": "all"},
            "combination": {"location": ["head"], "max_elements": 2},
        }
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
print(
    rows(
        {
            "coordinates": [
                {"activation": "resid.post", "layer": -2, "position": -1},
                {"activation": "mlp.out", "layer": -1, "position": -1},
            ]
        }
    )
)
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
