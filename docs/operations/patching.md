# Activation patching

Activation patching asks which activations cause a behaviour. The model runs on a target prompt; one
activation is overwritten with what it was on a source prompt (or with a value, or with noise); and the
change in the output measures how much that activation mattered. Every row of the index is a location
to patch, and gives one patched run.

| Method | Method config | Fields | How it obtains each row's effect |
|---|---|---|---|
| `exact` | `PatchingMethodConfig` | none | patches and runs the model, one run per row |
| `attribution` | `PatchingMethodConfig` | none | a first-order estimate from one forward and one backward pass for the whole index |
| `integrated_gradients` | `IntegratedGradientsConfig` | `interpolation_steps` (default `10`) | the same estimate, with the gradient averaged along the path from the source activation to the target one |

## A run

The classic sweep patches the residual stream of a clean prompt into a corrupted one, at every layer
and position:

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope import (
    MetricComparison,
    ModelAxis,
    ModelLayerLocation,
    Patcher,
    PatchingConfig,
    PatchingElementConfig,
    PatchingInput,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

residual = PatchingElementConfig(
    type=PatchingType.REPLACE,
    activation=ModelLayerLocation.RESIDUAL_PRE,
    location=[ModelAxis.LAYER, ModelAxis.POSITION],
)
config = PatchingConfig(
    metric={PatchingMetric.LOGIT_DIFF: [MetricComparison.NORMALIZED]},
    method=PatchingMethodConfig(type="exact"),
    patch=[SinglePatchingConfig(source=residual, target=residual)],
)
patching_input = PatchingInput(
    config,
    "The capital of France is",  # source
    "The capital of Spain is",  # target
    correct_token=" Paris",
    distractor_token=" Madrid",
)
Patcher(model).apply(patching_input, save_dir="runs/residual_sweep")
```

Each row of `result.csv` holds one patched run's `normalized_logit_diff`: 0 where the patch changed
nothing, 1 where the target run now prefers " Paris" over " Madrid" as much as the source run does.
For GPT-2 the grid shows the answer travelling. Up to layer 9, patching the `" France"` position (4)
alone restores it almost fully; by layers 10 and 11 the last position carries it instead:

```text
target_position    0    1    2    3     4     5
target_layer
0                0.0  0.0  0.0  0.0  1.00  0.00
1                0.0  0.0  0.0  0.0  1.00  0.00
2                0.0  0.0  0.0  0.0  1.00 -0.00
3                0.0  0.0  0.0  0.0  1.00  0.01
4                0.0  0.0  0.0  0.0  1.00  0.01
5                0.0  0.0  0.0  0.0  1.02  0.03
6                0.0  0.0  0.0  0.0  1.02  0.04
7                0.0  0.0  0.0  0.0  0.98  0.08
8                0.0  0.0  0.0  0.0  0.98  0.09
9                0.0  0.0  0.0  0.0  0.90  0.15
10               0.0  0.0  0.0  0.0  0.30  0.65
11               0.0  0.0  0.0  0.0  0.13  0.80
```

In a config file this is the `residual_sweep` entry of
[`examples/patching.yaml`](../examples.md#activation-patching),
which has an entry for every variant on this page.

## The two runs and the answer tokens

`PatchingInput(config, source_prompt, target_prompt, correct_token=..., distractor_token=...)`:

- The **target run** is the one patched, and every metric is read off the patched target run.
- The **source run** is what `replace` and `value_scale` read. A config whose sources only write a
  value or noise reads no source run, and `source_prompt` may then be `None` (see
  [Without a source prompt](#without-a-source-prompt)).
- `correct_token` and `distractor_token` are the `r` and `r'` of the metrics below: each a string that
  tokenizes to a single token, or a token id. Left unset, the correct token is what the source run
  predicts and the distractor what the target run predicts. For the prompts above GPT-2 predicts
  `" now"` and `" the"`, so name them. A pair that comes out as the same token is rejected when a
  difference metric is asked for, since every difference would be zero.

The metrics are read at the final position of each run, and the source and target prompts are each
run once and cached before the patching starts.

## Metrics and comparisons

`metric` maps each metric to the comparisons it is reported under, and every pair is a column of the
result. All of them come off the same patched run, so a second metric costs no extra forward pass:

```yaml
metric:
  logit_diff: [normalized, absolute]
  kl_divergence: [indirect_effect]
```

The metrics, read off a run's final-position logits:

| Metric | Value | Needs |
|---|---|---|
| `logit_diff` | $\mathrm{logit}(r) - \mathrm{logit}(r')$ | both tokens |
| `logprob_diff` | $\log P(r) - \log P(r')$, numerically the same as `logit_diff` | both tokens |
| `probability` | $P(r)$ | the correct token |
| `log_prob` | $\log P(r)$ | the correct token |
| `kl_divergence` | $D_{KL}(P_{\text{source}} \,\Vert\, P_{\text{run}})$, how far the run's whole distribution is from the source run's | a source run, no tokens |

The comparisons, where $m$ is a metric and the subscripts name the run it is read off:

| Comparison | Reported value | Column | Needs a source run |
|---|---|---|---|
| `absolute` | $m_{\text{patched}}$ | `logit_diff` | no |
| `indirect_effect` | $m_{\text{patched}} - m_{\text{target}}$ | `indirect_effect_logit_diff` | no |
| `normalized` | $(m_{\text{patched}} - m_{\text{target}}) / (m_{\text{source}} - m_{\text{target}})$ | `normalized_logit_diff` | yes |

`indirect_effect` is activation patching stated as a causal quantity. `normalized` puts 0 at "the
patch changed nothing" and 1 at "the patch reproduced the source run". This is the ratio circuit work
calls faithfulness, with the unpatched target playing the fully ablated model. A positive effect means
the patch moved the model toward the source run for every metric: a divergence falls as the run nears
the source, so `kl_divergence`'s indirect effect is its decrease.

The comparison is chosen per metric because the natural one differs: a divergence reads well as an
effect, a logit difference as a fraction of the gap between the two runs. A metric may list several.

## Patch elements

`patch` lists the elements to patch. An element is a `SinglePatchingConfig` with a `source` and a
`target`, each a `PatchingElementConfig`, which is a [locator](../concepts/locating.md) plus `type`
and `value`. Row `i` of the source index is read and written at row `i` of the target index. A side
with a single row pairs with every row of the other; otherwise the two need as many rows, and a pair
that does not line up is rejected before anything runs. The two sides may name different locations,
such as reading layer 10 and writing layer 11, as long as the slices fit.

The source's `type` says what gets written:

| Source `type` | Writes | Reads the source run |
|---|---|---|
| `replace` | the source run's activation | yes |
| `value_scale` | the source run's activation times `value` | yes |
| `value_replace` | the constant `value`; with the default `0.0` this is zero ablation | no |
| `gaussian_noise` | the target run's own activation plus Gaussian noise, the corruption of causal tracing | no |

The noise's standard deviation is three times the target run's own standard deviation over the whole
activation, as in causal tracing (Meng et al., 2022). It is drawn from a fixed seed, so every row is
corrupted by the same draw and a rerun repeats the sweep exactly. The target's `type` is always
`replace`. A type that cannot run, such as another target type or a source that reads a source run
the input has no prompt for, is rejected as the `PatchingInput` is built.

### Several elements at once

With more than one element, patched run `i` applies row `i` of every element together, so their
indices must have the same number of rows, or one. An element with one row is applied on every run.
The result is laid out along the element with the most rows, the first of them when several tie, so
every patched run has a row of its own. On the result page, a one-row element is named in the subtitle
and a varying one in every cell's hover.

```yaml
patch:
  - source: &late_heads  # varies: 4 layers x 12 heads
      activation: attn.z
      location: [layer, head]
      filter:
        - layer: {range: [8, 11]}
      type: replace
    target: *late_heads
  - source: &first_mlp  # one row, patched on every run
      activation: mlp.out
      location: [layer]
      filter:
        - layer: 0
      type: replace
    target: *first_mlp
```

## Methods

### `exact`

`exact` patches the model and runs it once per row. It assumes nothing about how the model responds,
at the price of a forward pass per row. `Patcher(model, batch_size=n)` runs `n` rows per forward pass.

Backends can round differently at different batch shapes (MPS does, by up to half a logit in
bfloat16). The source and target runs are therefore cached at the same batch width as the patched
runs, and a ragged last batch is padded, so a patch that writes back what was already there
reproduces the target run exactly.

`exact` also records what each patched run predicts next, as `generation` and `str_generation`.

### `attribution`

Attribution patching estimates every row's effect from the gradient of the metric at the target run's
activation: $\nabla m \cdot (a_{\text{patch}} - a_{\text{target}})$. One forward pass and one backward
pass per metric score the whole index, so the cost does not grow with the sweep. The estimate is only
as good as the model's local linearity, and a large intervention, such as zero-ablating a whole
layer, leaves that behind. The replacement comes from the same source types as `exact`, and a row
patching several elements sums their contributions.

### `integrated_gradients`

`integrated_gradients` averages that gradient over `interpolation_steps` points on the straight path
from the source run's activation to the target run's (Marks et al., 2024, Eq. 4), which recovers
effects a single gradient at the clean point hides. It costs one forward pass and one backward pass
per metric for every step, still flat in the size of the index. The path runs between the two runs,
so it needs a source prompt of the same token length as the target. It is meant for `replace`
sources: a value or noise is written off that path, and `attribution` is the estimate to use there.

The gradient methods make no patched run, so their results have no `generation`. A common workflow
is to estimate a large sweep with `attribution`, then confirm its strongest rows with `exact`.

## Without a source prompt

Zero ablation writes a value, so it reads no source run. The source names only its type, a single row
that pairs with every target row:

```python
from entoscope.util.data import from_dict

zero_mlp = from_dict(
    PatchingConfig,
    {
        "metric": {"logit_diff": ["indirect_effect"]},
        "method": "exact",
        "patch": [
            {
                "source": {"type": "value_replace"},
                "target": {"activation": "mlp.out", "location": ["layer"], "type": "replace"},
            }
        ],
    },
)
ablation = PatchingInput(zero_mlp, None, "The capital of Spain is", correct_token=" Paris", distractor_token=" Madrid")
Patcher(model).apply(ablation, save_dir="runs/zero_mlp")
```

Reported as an indirect effect, each row is $m(x \mid do(a = 0)) - m(x)$. Without a source run there
is no `normalized` and no `kl_divergence`, and the correct token has to be named.

## Output

`result.csv` has one row per patched run:

- the index of the element the result is laid out along, as `source_*` and `target_*` columns, a
  one-row source or target repeated on every row it pairs with;
- a column per metric and comparison pair, in the order the config names them: the metric's name for
  `absolute`, otherwise the comparison then the metric, as in `normalized_logit_diff`;
- `generation` and `str_generation`, the token the patched run predicts next, for `exact`.

Every element's index is written to its own `index-{i}.csv`.

`result.html` draws one grid per result column, on a diverging scale centred on what an ineffective
patch would score: the target run's own metric for `absolute`, zero otherwise. Warm cells moved the
model toward the source run. The subtitle names both prompts and what each predicts, and a cell's
hover gives its value, its patched run's prediction and its full source and target coordinates.

## Recipes

All of these are entries of
[`examples/patching.yaml`](../examples.md#activation-patching):

| Question | Entry |
|---|---|
| Where, layer by layer and token by token, is the answer carried? | `residual_sweep` |
| Which heads matter? | `head_sweep`: `attn.z` over `[layer, head]` |
| Which MLP layers matter? | `zero_ablation`: `value_replace` into `mlp.out` over `[layer]` |
| Which tokens matter? | `noise_tracing`: `gaussian_noise` on `embed` over `[position]` |
| A whole sweep, cheaply | `attribution`, `integrated_gradients` |
| Specific points | `explicit_coordinates` |
| A sweep on top of a fixed patch | `heads_with_fixed_mlp` |

## References

- Kevin Meng, David Bau, Alex Andonian and Yonatan Belinkov. [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). 2022.
- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris and Jacob Steinhardt. [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593). 2022.
- Fred Zhang and Neel Nanda. [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042). 2023.
- Stefan Heimersheim and Neel Nanda. [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255). 2024.
- Neel Nanda. [Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). Blog post.
- Aaquib Syed, Can Rager and Arthur Conmy. [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348). 2023.
- Mukund Sundararajan, Ankur Taly and Qiqi Yan. [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365). 2017.
- Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau and Aaron Mueller. [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647). 2024.
