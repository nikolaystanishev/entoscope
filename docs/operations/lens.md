# Lens

A lens reads a location in the model out onto the vocabulary and reports the tokens it ranks highest.
The logit and tuned lenses read a prompt's activations, and ask what the model would predict if it
stopped there. The parameter lens reads the weights, and asks which tokens a neuron or a head reads or
writes.

| Method | Reads | Method config | Fields |
|---|---|---|---|
| `logit_lens` | activations | `LensMethodConfig` | none |
| `tuned_lens` | activations, through trained translators | `TunedLensConfig` | `translators` |
| `parameter_lens` | weights, no prompt | `ParameterLensConfig` | `projection`, `scale` |

## A run

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import (
    Filter,
    Lens,
    LensConfig,
    LensInput,
    LensMethodConfig,
    ModelAxis,
    ModelLayerLocation,
    ModelLocatorConfig,
)

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

# What every head of every layer writes into the residual stream at the last position.
config = LensConfig(
    method=LensMethodConfig(type="logit_lens"),
    top_k=5,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.POSITION],
        filter=Filter.from_config([{"position": -1}]),
    ),
)
Lens(model).apply(LensInput(config, "The capital of France is"), save_dir="runs/heads")
```

The same config in a file is the `head_logit_lens` entry of
[`examples/lens.yaml`](../examples.md#lens).

### Config and input

`LensConfig` fields:

| Field | Meaning | Default |
|---|---|---|
| `method` | the lens, as its method config (above) | required |
| `top_k` | how many tokens each row keeps, at least 1 | `1` |
| `source` | where the lens reads, a [locator](../concepts/locating.md) | required |

`LensInput(config, prompt)` pairs the config with its prompt. The prompt is `None` exactly when the
lens reads weights, and either mismatch is rejected when the input is built.

Each row is ranked as a single vector over the vocabulary, so a row reads one layer and, for the
activation lenses, one position: keep `position` in `location` or pin it in a filter or coordinate. A
row may still group heads or neurons, which a lens sums into one direction.

## Logit lens

The logit lens (nostalgebraist, 2020) reads an element the way the model reads its own final residual
stream, `ln_final(x) @ W_U + b_U`. What `x` is depends on the activation:

| Activation | `x` |
|---|---|
| `embed`, `resid.pre`, `resid.mid`, `resid.post`, `attn.out`, `mlp.out` | the activation itself |
| `attn.in`, `mlp.in` | the activation itself: the residual stream after the norm before the sublayer, as the sublayer reads it |
| `attn.out_pre_norm`, `mlp.out_pre_norm` | the activation itself: what the sublayer writes before the norm after it, the same tensor as `attn.out` and `mlp.out` on a model without one |
| `mlp.pre`, `mlp.pre_linear`, `mlp.post` | the activation written into the residual stream through the layer's `W_out`: what the indexed neurons write |
| `attn.v`, `attn.z` | the activation written through the head's `W_O`: what the indexed heads write |

Queries and keys are read differently, against the embedding through the other side of their head. A
query is mapped through `W_K` and a key through `W_Q`, then scored against `W_E` and divided by the
attention scale. A query's tokens are those whose embeddings, read as keys, it attends to most, and
the other way around for a key. `attn.pattern` cannot be read out onto the vocabulary.

On a model with grouped-query attention, `attn.k` and `attn.v` have fewer heads than the weights that
would read them out, and the lens rejects them.

## Tuned lens

The tuned lens (Belrose et al., 2023) moves each residual state into the basis of the last layer
with a trained affine translator before reading it out as the logit lens does. Early layers are then
read in the space the model itself ends up in.

`TunedLensConfig` has one field. `translators` names the pretrained translators, either as a resource
id on the tuned-lens hub space or as a local directory holding `config.json` and `params.pt`. Left
unset, the model's tokenizer name is used, or its model name when it has no tokenizer name; for GPT-2
that finds the pretrained translators:

```python
from entoscope import TunedLensConfig

tuned = LensConfig(
    method=TunedLensConfig(type="tuned_lens"),
    top_k=5,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
    ),
)
Lens(model).apply(LensInput(tuned, "The capital of France is"), save_dir="runs/tuned")
```

The translators are downloaded and checked against the model's width and depth on first use, not when
the lens is built, so planning a tuned-lens run downloads nothing. How each activation is translated:

- The translators are indexed by residual state: state `l` is `resid.pre` of layer `l`, and what layer
  `l` writes lands in state `l + 1`. The last state is the one the model reads out itself, so it is
  not translated.
- A residual state (`resid.pre`, `resid.mid`, `resid.post`) goes through its whole translator.
- A component of a state (`embed`, `attn.out`, `mlp.out`, and the MLP and head activations) takes only
  its state's linear part, since the bias belongs to the state as a whole. `embed` is a component of
  state 0.
- Queries and keys cannot be read: their translation would need the head's other side.
- A sublayer's own input or output (`attn.in`, `mlp.in`, `attn.out_pre_norm`, `mlp.out_pre_norm`)
  cannot be read either: the translators were trained on the residual states and what the sublayers
  add to them, which a sublayer's input is not, nor its output before a post-norm. The logit lens reads
  them.

## Parameter lens

The parameter lens (Dar et al., 2022) reads the model's weights in vocabulary space: a `d_model`
weight vector projected onto the vocabulary directly, with no LayerNorm and no bias. Nothing about it
depends on an input, so the run takes no prompt:

```python
from entoscope import ParameterLensConfig

# The value vector neuron 373 writes into the residual stream, in every layer.
values = LensConfig(
    method=ParameterLensConfig(type="parameter_lens"),
    top_k=5,
    source=ModelLocatorConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelAxis.LAYER, ModelAxis.NEURON],
        filter=Filter.from_config([{"neuron": 373}]),
    ),
)
Lens(model).apply(LensInput(values), save_dir="runs/values")
```

Each activation names a weight vector:

| Activation | Weight vector | Reading |
|---|---|---|
| `mlp.pre` | a column of `W_in` (`W_gate` on a gated MLP) | the key the neuron reads with |
| `mlp.pre_linear` | a column of `W_in` (gated MLPs only) | the key of the linear branch |
| `mlp.post` | a row of `W_out` | the value the neuron writes |
| `attn.q`, `attn.k`, `attn.v` | a column of `W_Q`, `W_K`, `W_V` | what a head dimension reads |
| `attn.z` | a row of `W_O` | what a head dimension writes |

Attention weights are read one head dimension at a time (`location: [layer, head, neuron]`). A row
needs a single layer, since every weight lives in one block. A list of neurons or heads sums their
vectors into one direction, and leaving an axis out of the location sums all of its vectors: `attn.z`
over `[layer, head]` reads each head's whole `W_O`. `position` is not a location of this lens, and the
residual-stream activations name no weight vector.

`ParameterLensConfig` fields:

| Field | Meaning | Default |
|---|---|---|
| `projection` | `unembedding` multiplies by `W_U`, `embedding` by `W_E` transposed, as in the paper | `unembedding` |
| `scale` | multiplies the vector before projecting it | `1.0` |

The two projections differ whenever a model does not tie its embedding and unembedding. GPT-2 ties
them, so on HuggingFace's weights they agree exactly, until `enable_compatibility_mode` folds
LayerNorm in and centers the unembedding. A negative `scale` such as `-1.0` reads a vector against its
own direction, surfacing the tokens a weight writes against rather than toward; its magnitude acts as
a temperature on the probabilities.

The probabilities are a softmax over the projected vector. They rank the vocabulary, but they are not
a distribution the model ever produces.

## Output

`result.csv` has one row per index row: its location, then for each rank `i` from 0 (the best) to
`top_k - 1`:

| Column | Meaning |
|---|---|
| `token_{i}` | the token id |
| `str_token_{i}` | the token as text, with non-printable characters escaped |
| `probability_{i}` | its softmax probability over the row's logits |

`result.html` holds one grid per rank. A cell's colour is that rank's probability on a fixed 0 to 1
scale, the cell is labelled with the rank's token (up to 600 cells a grid), and the hover lists the
row's whole top-k. See [Running and reading results](../concepts/running.md) for the files and the
grid's layout.

## References

- nostalgebraist. [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong, 2020.
- Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman
  and Jacob Steinhardt. [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). 2023.
- Guy Dar, Mor Geva, Ankit Gupta and Jonathan Berant. [Analyzing Transformers in Embedding Space](https://arxiv.org/abs/2209.02535). 2022.
