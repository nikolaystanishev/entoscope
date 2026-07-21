# intex

A mechanistic-interpretability toolkit for large language models, built on
[`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens). Each operation follows
the same shape — `Config → Input → Operator → Collector` — and can be driven inline or from an
experiment script. Current operations: Logit Lens, Probing, Activation Patching, and Circuit
Discovery.

## Circuit Discovery

Discovers the circuit a model uses on a set of prompts. It supports multiple methods; the first is
**Information Flow Routes**, which builds a contribution graph over the residual stream and prunes the
routes that lead to the model's prediction.

```python
from transformer_lens import HookedTransformer

from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.circuit.discoverer import CircuitDiscoverer
from intex.circuit.input import CircuitInput

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

config = CircuitConfig(type=CircuitMethodType.INFORMATION_FLOW_ROUTES, threshold=0.04)
prompts = ["The capital of France is", "The Eiffel Tower is in"]

collector = CircuitDiscoverer(model).apply(
    CircuitInput(prompts, config), save_dir="runs/circuit", return_collector=True
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
`CircuitDiscoverer(model, collector_class=CircuitCollector)`.

Configs can also be loaded from YAML via `intex.operation.config.ConfigFactory[CircuitConfig]`, and
the operation can be run from an `@intex.script.experiment.experiment` script.

## Attribution & licensing

The Information Flow Routes method reuses contribution-graph math from the
[LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool)
(`src/intex/circuit/vendor/contributions.py`, vendored — see `src/intex/circuit/vendor/NOTICE`),
which is licensed **CC BY-NC 4.0**. Because that code is non-commercial, the Circuit Discovery
`information_flow_routes` method — and any results produced with it — may be used for
**non-commercial purposes only**. Method reference:

> Javier Ferrando and Elena Voita. "Information Flow Routes: Automatically Interpreting Language
> Models at Scale." 2024. https://arxiv.org/abs/2403.00824
