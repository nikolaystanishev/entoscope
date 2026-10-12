# Circuit discovery

Circuit discovery finds the routes through the model that carry its prediction on a prompt. A circuit
spans every layer, position and head, so there is no locator: the config is the method alone, and a
run builds one graph per prompt.

| Method | Method config | Fields |
|---|---|---|
| `information_flow_routes` | `InformationFlowRoutesConfig` | `threshold` (default `0.04`), `path_threshold` (default: `threshold`), `prediction_positions` (default `[-1]`) |

`information_flow_routes` builds on code licensed CC BY-NC 4.0, so it and the results it produces
may be used for non-commercial purposes only (see [Licensing](#licensing)).

## A run

```python
from transformer_lens.model_bridge import TransformerBridge

from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutesConfig

model = TransformerBridge.boot_transformers("gpt2")
model.eval()

config = CircuitConfig(method=InformationFlowRoutesConfig(type="information_flow_routes", threshold=0.04))
prompts = ["The capital of France is", "The Eiffel Tower is in"]

collector = CircuitDiscoverer(model).apply(
    CircuitInput(config, prompts), save_dir="runs/circuit", return_collector=True
)

first = collector.result[0]  # a PromptCircuit per prompt
print(first.graph.number_of_edges(), first.routes.number_of_edges())
```

`CircuitInput(config, prompts)` takes at least one prompt. `collector.result` is a list with one
`PromptCircuit` per prompt, holding the prompt, its `str_tokens`, the full `graph` and the pruned
`routes`, both `networkx.MultiDiGraph`.

## Information Flow Routes

Information Flow Routes (Ferrando and Voita, 2024) runs the prompt once and measures how much every
input to each residual-stream representation contributes to it, layer by layer. In each layer it
measures every head's contribution from every earlier token, the residual connection around attention,
the MLP and the residual connection around the MLP. Contributions below `threshold` are dropped and the
rest renormalized, so the incoming weights of every representation still sum to 1; `0.0` keeps every
edge. The result is the full contribution graph.

The routes are the part of that graph the prediction depends on. Starting from the last layer's
representation at each of `prediction_positions` (negative positions count from the prompt's end, and
`[-1]` is the next-token prediction), the method walks backward over the edges heavier than
`path_threshold` and keeps every edge it reaches. With several positions, their routes are merged into
one graph.

The method assumes each block runs attention and then the MLP, one after the other. Models that run
them in parallel (`cfg.parallel_attn_mlp`, such as GPT-NeoX with its parallel residual) are rejected.
Grouped-query attention is supported.

## The graph

Nodes are named `{prefix}{layer}_{position}`:

| Prefix | `kind` | The residual stream |
|---|---|---|
| `X` | `input` | entering the first block, always at layer 0 (`X0_3` is token 3's input) |
| `A` | `after_attn` | after layer `l`'s attention |
| `M` | `mlp` | layer `l`'s MLP, between `A` and `I` |
| `I` | `after_ffn` | after layer `l`'s MLP, which is the input of layer `l + 1` |

Every node carries `kind`, `layer`, `position` and `token`. Edges carry `kind`, `weight` and `head`, and
the graph keeps a separate attention edge per head:

| Edge `kind` | Key | From, to |
|---|---|---|
| `attention` | `h{head}` | the source token's previous node (`X0_` or `I{l-1}_`) to the destination token's `A{l}_` |
| `residual` | `residual` | the token's previous node to `A{l}_`, and `A{l}_` to `I{l}_` |
| `mlp` | `mlp` | `A{l}_` to `M{l}_` and `M{l}_` to `I{l}_`, both with the MLP's weight |

`head` is `-1` on the edges that are not attention.

## Output

For each prompt `i`, the run writes:

- `graph_{i}.graphml` and `graph_{i}.json`: the full graph, as GraphML and as networkx node-link JSON.
- `routes_{i}.graphml` and `routes_{i}.json`: the routes to the prediction.
- `circuit_{i}.html`: the interactive page.

`prompts.json` lists every prompt with its tokens and both graphs' edge counts, and `config.json`
holds the config. The node-link JSON reads back into the same multigraph:

```python
import json
from pathlib import Path

import networkx as nx

routes = nx.node_link_graph(json.loads(Path("runs/circuit/routes_0.json").read_text()), edges="links")
print(routes.number_of_edges() == first.routes.number_of_edges())  # True
```

The page holds two plots: the routes to the prediction, then the full graph. Token positions run
across and layers up, with the inputs at the bottom. Attention edges are coloured by head and sized by
weight; residual edges are dotted and MLP edges orange. Hovering an edge's marker gives its endpoints
and weight, and for an attention edge its layer and head.

## A method of your own

A registered circuit method returns a `PromptCircuit` per prompt (see
[Custom methods](../guides/custom-methods.md)). The page reads the node and edge attributes above, so
a method that writes other attributes runs with `collector_class=CircuitCollector`, which writes the
graph files and no page.

## Licensing

The contribution math of `information_flow_routes` is vendored from the
[LM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool), licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The method, and any results produced
with it, may therefore be used for non-commercial purposes only. `src/entoscope/circuit/vendor/NOTICE`
records the source, the commit and the changes made.

## References

- Javier Ferrando and Elena Voita. [Information Flow Routes: Automatically Interpreting Language Models at Scale](https://arxiv.org/abs/2403.00824). 2024.
