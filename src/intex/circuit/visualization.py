"""Interactive Plotly rendering of a discovered circuit.

Lays the graph out as the Information Flow Routes figures do: token position on the x axis, model
depth on the y axis (input tokens at the bottom, the final layer at the top). Attention edges are
drawn per head — the coloured midpoint markers encode which head carries each edge (colour) and how
strong it is (size). One self-contained HTML file is produced per prompt, holding two figures: the
pruned routes-to-prediction (headline) and the full contribution graph.
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go

from intex.circuit.method.circuit import PromptCircuit

_KIND_COLOR = {
    "input": "#111827",
    "after_attn": "#2563eb",
    "mlp": "#f59e0b",
    "after_ffn": "#111827",
}
_ATTENTION_LINE = "rgba(120,120,120,0.35)"
_RESIDUAL_LINE = "rgba(160,160,160,0.30)"
_MLP_LINE = "rgba(245,158,11,0.45)"


def _node_position(attrs: dict) -> tuple[float, float]:
    kind, layer, position = attrs["kind"], attrs["layer"], attrs["position"]
    if kind == "input":
        return position, 0.0
    if kind == "after_attn":
        return position, 2 * layer + 1
    if kind == "mlp":
        return position + 0.30, 2 * layer + 1.5
    # after_ffn
    return position, 2 * layer + 2


def _marker_size(weight: float) -> float:
    return 6.0 + 22.0 * min(max(weight, 0.0), 1.0)


def _line_trace(segments: list[tuple[float, float]], color: str, name: str, dash: str | None = None) -> go.Scatter:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for (x0, y0), (x1, y1) in zip(segments[::2], segments[1::2], strict=True):
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=1.2, dash=dash),
        hoverinfo="skip",
        name=name,
        showlegend=False,
    )


def _figure(graph: nx.MultiDiGraph, str_tokens: list[str], n_layers: int, title: str) -> go.Figure:
    fig = go.Figure()

    if graph.number_of_nodes() == 0:
        fig.add_annotation(text="No edges above threshold", showarrow=False, font=dict(size=16))
        fig.update_layout(title=title, template="plotly_white")
        return fig

    pos = {node: _node_position(attrs) for node, attrs in graph.nodes(data=True)}

    attn_lines: list[tuple[float, float]] = []
    resid_lines: list[tuple[float, float]] = []
    mlp_lines: list[tuple[float, float]] = []

    attn_mx, attn_my, attn_head, attn_size, attn_hover = [], [], [], [], []
    other_mx, other_my, other_color, other_size, other_hover = [], [], [], [], []

    for u, v, data in graph.edges(data=True):
        (x0, y0), (x1, y1) = pos[u], pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        kind, weight, head = data["kind"], data["weight"], data["head"]

        if kind == "attention":
            attn_lines += [(x0, y0), (x1, y1)]
            attn_mx.append(mx)
            attn_my.append(my)
            attn_head.append(head)
            attn_size.append(_marker_size(weight))
            attn_hover.append(
                f"attention · L{graph.nodes[v]['layer']} · head {head}<br>{u} → {v}<br>weight {weight:.3f}"
            )
        else:
            (resid_lines if kind == "residual" else mlp_lines).extend([(x0, y0), (x1, y1)])
            other_mx.append(mx)
            other_my.append(my)
            other_color.append(_RESIDUAL_LINE if kind == "residual" else _MLP_LINE)
            other_size.append(_marker_size(weight))
            other_hover.append(f"{kind}<br>{u} → {v}<br>weight {weight:.3f}")

    if resid_lines:
        fig.add_trace(_line_trace(resid_lines, _RESIDUAL_LINE, "residual", dash="dot"))
    if mlp_lines:
        fig.add_trace(_line_trace(mlp_lines, _MLP_LINE, "mlp"))
    if attn_lines:
        fig.add_trace(_line_trace(attn_lines, _ATTENTION_LINE, "attention"))

    if other_mx:
        fig.add_trace(
            go.Scatter(
                x=other_mx,
                y=other_my,
                mode="markers",
                marker=dict(color=other_color, size=other_size, symbol="square", line=dict(width=0)),
                text=other_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    if attn_mx:
        fig.add_trace(
            go.Scatter(
                x=attn_mx,
                y=attn_my,
                mode="markers",
                marker=dict(
                    color=attn_head,
                    size=attn_size,
                    colorscale="Turbo",
                    cmin=0,
                    cmax=max(attn_head),
                    colorbar=dict(title="head"),
                    line=dict(width=0.5, color="white"),
                ),
                text=attn_hover,
                hoverinfo="text",
                name="attention head",
                showlegend=False,
            )
        )

    node_x, node_y, node_color, node_hover = [], [], [], []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(_KIND_COLOR.get(attrs["kind"], "#111827"))
        node_hover.append(
            f"{node}<br>{attrs['kind']} · L{attrs['layer']} · pos {attrs['position']}<br>'{attrs['token']}'"
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(color=node_color, size=6, symbol="circle"),
            text=node_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    y_ticks = [0] + [2 * layer + 2 for layer in range(n_layers)]
    y_text = ["input"] + [f"L{layer}" for layer in range(n_layers)]
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(400, 55 * n_layers + 150),
        width=max(600, 130 * len(str_tokens) + 200),
        hovermode="closest",
        margin=dict(l=70, r=40, t=70, b=70),
        xaxis=dict(
            title="token position",
            tickmode="array",
            tickvals=list(range(len(str_tokens))),
            ticktext=[f"{i}: {tok}" for i, tok in enumerate(str_tokens)],
        ),
        yaxis=dict(title="layer", tickmode="array", tickvals=y_ticks, ticktext=y_text),
    )
    return fig


def render_circuit_html(circuit: PromptCircuit, n_layers: int, path: str) -> None:
    routes_fig = _figure(circuit.routes, circuit.str_tokens, n_layers, f"Information Flow Routes — {circuit.prompt!r}")
    full_fig = _figure(circuit.graph, circuit.str_tokens, n_layers, f"Full contribution graph — {circuit.prompt!r}")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        f"<title>Circuit — {circuit.prompt}</title></head><body>",
        routes_fig.to_html(full_html=False, include_plotlyjs=True),
        "<hr/>",
        full_fig.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
