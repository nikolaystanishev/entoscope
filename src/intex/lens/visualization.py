"""Interactive Plotly rendering of a lens run.

The grid is the run's own index — layer down, position (or head, or neuron) across — coloured by the
top-1 probability, with the top-1 token printed in the cell and the whole top-k in hover.
"""

from __future__ import annotations

from html import escape

from intex.lens.input import LensInput
from intex.model.token import str_token_labels
from intex.operation.visualization import GridFigure, describe_row, format_value, titled


class LensFigure(GridFigure[LensInput]):
    @property
    def title(self) -> str:
        return titled(f"Lens — {self.operation_input.config.type.value}", self.operation_input.config)

    @property
    def subtitle(self) -> str:
        prompt = self.operation_input.prompt
        read = f'prompt: "{escape(prompt)}"' if prompt is not None else "weights (no prompt)"
        return f"{read} · top-1 of top-{self.operation_input.config.top_k}"

    @property
    def value_column(self) -> str:
        return "probability_0"

    @property
    def value_title(self) -> str:
        return "probability"

    def bar_yaxis(self) -> dict:
        return dict(title=self.value_title, range=[0.0, 1.05])

    def cell_text(self) -> list[str]:
        return [format_value(token) for token in self.result["str_token_0"]]

    def str_tokens(self) -> dict[str, list[str]]:
        # The parameter lens reads the weights, so it has no prompt and no tokens to label an axis with.
        if self.operation_input.prompt is None:
            return {}
        return {"source_position": str_token_labels(self.model, self.operation_input.tokens)}

    def hover(self, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        lines = [coordinate, describe_row(row, "source_")]
        for i in range(self.operation_input.config.top_k):
            token, probability = f"str_token_{i}", f"probability_{i}"
            if token in row.index and probability in row.index:
                lines.append(f"{i + 1}. {format_value(row[token])} — {format_value(row[probability])}")
        return "<br>".join(line for line in lines if line)
