"""Information Flow Routes circuit-discovery method.

Builds a per-head contribution graph over a single forward pass and prunes the routes that lead to
the model's prediction, following Ferrando & Voita, "Information Flow Routes: Automatically
Interpreting Language Models at Scale" (2024, https://arxiv.org/abs/2403.00824).

The node scheme (``X0_``/``A``/``M``/``I``) and the per-layer contribution loop follow the LM
Transparency Tool's ``llm_transparency_tool/routes/graph.py``, but the graph here is a
``networkx.MultiDiGraph`` so that each attention head is its own edge (the upstream ``GraphBuilder``
uses a ``DiGraph`` and sums attention over heads). Route pruning is reimplemented to work on a
multigraph (the upstream helper is ``DiGraph``-only — its ``filter_edge`` lambda takes two args,
which breaks for multigraphs).

The numerical contributions come from the vendored ``intex.circuit.vendor.contributions`` module,
copied from the LM Transparency Tool under CC BY-NC 4.0 (see ``intex/circuit/vendor/NOTICE``).
Because that code is non-commercial, this method — and any results it produces — may be used for
non-commercial purposes only.
"""

from __future__ import annotations

from enum import Enum

import einops
import networkx as nx
import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from intex.circuit.method.circuit import AbstractCircuitMethod, PromptCircuit
from intex.circuit.vendor import contributions
from intex.model import Model
from intex.model.config import ModelLayerLocation
from intex.model.index import TRANSFORMER_ACT_NAME
from intex.model.token import str_token_labels
from intex.model.weight import attention_out_weight

# Sentinel head id for edges that are not attention (residual skips and the MLP block).
NON_ATTENTION_HEAD = -1


class NodeType(Enum):
    """A residual-stream node component: its node-name prefix and its ``kind`` attribute value.

    Node names are ``{prefix}{layer}_{token}``; ``INPUT`` is always built at layer 0, so it renders as
    ``X0_{token}``.
    """

    INPUT = ("X", "input")
    AFTER_ATTENTION = ("A", "after_attn")
    MLP = ("M", "mlp")
    AFTER_FFN = ("I", "after_ffn")

    def __init__(self, prefix: str, kind: str):
        self.prefix = prefix
        self.kind = kind

    def node_name(self, layer: int, token: int) -> str:
        return f"{self.prefix}{layer}_{token}"


class _GraphBuilder:
    def __init__(self, n_layers: int, n_tokens: int, str_tokens: list[str]):
        self.n_layers = n_layers
        self.n_tokens = n_tokens
        self.graph = nx.MultiDiGraph()

        for token in range(n_tokens):
            self._add_node(NodeType.INPUT, 0, token, str_tokens[token])
        for layer in range(n_layers):
            for token in range(n_tokens):
                label = str_tokens[token]
                self._add_node(NodeType.AFTER_ATTENTION, layer, token, label)
                self._add_node(NodeType.MLP, layer, token, label)
                self._add_node(NodeType.AFTER_FFN, layer, token, label)

    def _add_node(self, node_type: NodeType, layer: int, position: int, label: str) -> None:
        self.graph.add_node(
            node_type.node_name(layer, position),
            kind=node_type.kind,
            layer=layer,
            position=position,
            token=label,
        )

    @staticmethod
    def _source(layer: int, token: int) -> str:
        if layer > 0:
            return NodeType.AFTER_FFN.node_name(layer - 1, token)
        return NodeType.INPUT.node_name(0, token)

    def output_node(self, token: int) -> str:
        return NodeType.AFTER_FFN.node_name(self.n_layers - 1, token)

    def add_attention_edge(self, layer: int, token_from: int, token_to: int, head: int, w: float) -> None:
        self.graph.add_edge(
            self._source(layer, token_from),
            NodeType.AFTER_ATTENTION.node_name(layer, token_to),
            key=f"h{head}",
            weight=w,
            kind="attention",
            head=head,
        )

    def add_residual_to_attn(self, layer: int, token: int, w: float) -> None:
        self.graph.add_edge(
            self._source(layer, token),
            NodeType.AFTER_ATTENTION.node_name(layer, token),
            key="residual",
            weight=w,
            kind="residual",
            head=NON_ATTENTION_HEAD,
        )

    def add_ffn_edge(self, layer: int, token: int, w: float) -> None:
        a = NodeType.AFTER_ATTENTION.node_name(layer, token)
        m = NodeType.MLP.node_name(layer, token)
        i = NodeType.AFTER_FFN.node_name(layer, token)
        self.graph.add_edge(a, m, key="mlp", weight=w, kind="mlp", head=NON_ATTENTION_HEAD)
        self.graph.add_edge(m, i, key="mlp", weight=w, kind="mlp", head=NON_ATTENTION_HEAD)

    def add_residual_to_ffn(self, layer: int, token: int, w: float) -> None:
        self.graph.add_edge(
            NodeType.AFTER_ATTENTION.node_name(layer, token),
            NodeType.AFTER_FFN.node_name(layer, token),
            key="residual",
            weight=w,
            kind="residual",
            head=NON_ATTENTION_HEAD,
        )


@torch.no_grad()
def decomposed_attn(
    model: Model,
    cache: ActivationCache,
    layer: int,
    batch_i: int = 0,
) -> Float[torch.Tensor, "pos key_pos head d_model"]:
    """Per (query_pos, key_pos, head) attention contribution to the residual stream, in d_model space.

    Reconstructs each head's output written to the residual stream, split by the key position it came
    from, before the residual add (no layernorm is applied — the attention output is added to the
    residual stream unnormalized). Summing over ``key_pos`` and ``head`` recovers ``attn_out`` minus
    the ``b_O`` bias. Handles grouped-query attention by repeating the value heads across their group.
    """
    # ``hook_v`` already carries ``b_V`` on both backends, so the bias is not added again here.
    hook_v = cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.ATTENTION_V], layer][batch_i]  # [key_pos, head_kv, d_head]
    v = hook_v.repeat_interleave(model.cfg.n_heads // hook_v.shape[-2], dim=-2)  # [key_pos, head, d_head]

    pattern = cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.ATTENTION_PATTERN], layer][batch_i].to(
        v.dtype
    )  # [head, query_pos, key_pos]

    z = einops.einsum(
        v,
        pattern,
        "key_pos head d_head, head query_pos key_pos -> query_pos key_pos head d_head",
    )
    return einops.einsum(
        z,
        attention_out_weight(model, layer),  # [head, d_head, d_model]
        "query_pos key_pos head d_head, head d_head d_model -> query_pos key_pos head d_model",
    )


@torch.no_grad()
def build_information_flow_graph(
    model: Model,
    cache: ActivationCache,
    n_tokens: int,
    str_tokens: list[str],
    threshold: float,
) -> nx.MultiDiGraph:
    """Build the full per-head contribution graph for one cached forward pass (batch item 0)."""
    if model.cfg.parallel_attn_mlp:
        raise NotImplementedError(
            "Information Flow Routes assumes a sequential attn -> resid_mid -> MLP block; parallel "
            "attention/MLP architectures (model.cfg.parallel_attn_mlp) are not supported."
        )
    n_layers = model.cfg.n_layers
    builder = _GraphBuilder(n_layers, n_tokens, str_tokens)

    for layer in range(n_layers):
        c_attn, c_resid_attn = contributions.get_attention_contributions(
            resid_pre=cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.RESIDUAL_PRE], layer],
            resid_mid=cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.RESIDUAL_MID], layer],
            decomposed_attn=decomposed_attn(model, cache, layer).unsqueeze(0),
        )
        c_attn, c_resid_attn = contributions.apply_threshold_and_renormalize(threshold, c_attn, c_resid_attn)

        n_heads = c_attn.shape[-1]
        for token_to in range(n_tokens):
            for token_from in range(n_tokens):
                for head in range(n_heads):
                    w = c_attn[0, token_to, token_from, head].item()
                    if w > 0:
                        builder.add_attention_edge(layer, token_from, token_to, head, w)
            resid_w = c_resid_attn[0, token_to].item()
            if resid_w > 0:
                builder.add_residual_to_attn(layer, token_to, resid_w)

        c_ffn, c_resid_ffn = contributions.get_mlp_contributions(
            resid_mid=cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.RESIDUAL_MID], layer],
            resid_post=cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.RESIDUAL_POST], layer],
            mlp_out=cache[TRANSFORMER_ACT_NAME[ModelLayerLocation.MLP_OUT], layer],
        )
        # apply_threshold_and_renormalize renormalizes over the block dims via
        # ``c_blocks.sum(dim=range(resid_dims, block_dims))``. Here c_ffn and c_resid_ffn have equal
        # rank, so that range is empty and torch's ``sum(dim=())`` reduces over ALL dims (torch, unlike
        # numpy, treats an empty dim tuple as a full reduction), which couples positions and breaks the
        # per-position renormalization. Add a trailing singleton block dim so the reduction stays
        # per-position (summing a size-1 dim is a no-op). The attention call above is unaffected — its
        # c_attn already has extra (key_pos, head) block dims.
        c_ffn, c_resid_ffn = contributions.apply_threshold_and_renormalize(threshold, c_ffn.unsqueeze(-1), c_resid_ffn)
        c_ffn = c_ffn.squeeze(-1)

        for token in range(n_tokens):
            ffn_w = c_ffn[0, token].item()
            if ffn_w > 0:
                builder.add_ffn_edge(layer, token, ffn_w)
            resid_w = c_resid_ffn[0, token].item()
            if resid_w > 0:
                builder.add_residual_to_ffn(layer, token, resid_w)

    return builder.graph


def resolve_positions(positions: list[int], n_tokens: int) -> list[int]:
    """Turn possibly-negative prediction positions into distinct in-range indices, order preserved."""
    resolved: list[int] = []
    for position in positions:
        index = position + n_tokens if position < 0 else position
        if not 0 <= index < n_tokens:
            raise ValueError(f"Prediction position {position} is out of range for a {n_tokens}-token prompt")
        if index not in resolved:
            resolved.append(index)
    return resolved


def prune_routes_to_predictions(
    graph: nx.MultiDiGraph,
    n_layers: int,
    starting_positions: list[int],
    threshold: float,
) -> nx.MultiDiGraph:
    """Keep only the edges on paths that reach the output node(s), traced backwards over strong edges.

    Mirrors the upstream ``build_paths_to_predictions`` (reverse the graph, restrict to edges above
    ``threshold``, DFS back from each output node) but operates on a ``MultiDiGraph`` and merges the
    per-position trees into a single returned graph.
    """
    reversed_graph = graph.reverse(copy=False)

    def keep_edge(u: str, v: str, key: str) -> bool:
        return reversed_graph[u][v][key]["weight"] > threshold

    search = nx.subgraph_view(reversed_graph, filter_edge=keep_edge)

    edges: set[tuple[str, str, str]] = set()
    for start in starting_positions:
        source = NodeType.AFTER_FFN.node_name(n_layers - 1, start)
        if source not in search:
            continue
        edges.update(nx.edge_dfs(search, source=source))

    if not edges:
        return nx.MultiDiGraph()

    # edge_subgraph returns a view over the reversed graph; copy it and flip edges back to forward.
    return nx.MultiDiGraph(search.edge_subgraph(edges).reverse(copy=True))


class InformationFlowRoutes(AbstractCircuitMethod):
    @torch.no_grad()
    def build(self, prompt: str, tokens: Int[torch.Tensor, "batch pos"]) -> PromptCircuit:
        _, cache = self.model.run_with_cache(tokens)

        n_tokens = tokens.shape[1]
        str_tokens = str_token_labels(self.model, tokens)

        graph = build_information_flow_graph(self.model, cache, n_tokens, str_tokens, self.config.threshold)
        routes = prune_routes_to_predictions(
            graph,
            self.model.cfg.n_layers,
            resolve_positions(self.config.prediction_positions, n_tokens),
            self.config.resolved_path_threshold(),
        )

        return PromptCircuit(prompt=prompt, str_tokens=str_tokens, graph=graph, routes=routes)
