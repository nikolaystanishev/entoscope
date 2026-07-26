"""Interactive Plotly rendering of a lens run.

The grid is the run's own index — layer down, position (or head, or neuron) across — coloured by the
top-1 probability, with the top-1 token printed in the cell and the whole top-k in hover.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from intex.lens.input import LensInput
from intex.operation.visualization import GridFigure, describe_row, format_value, titled


class LensFigure(GridFigure):
    def __init__(self, result: pd.DataFrame, lens_input: LensInput, tokens: dict[str, list[str]]) -> None:
        super().__init__(result, tokens)
        self.lens_input = lens_input

    @property
    def title(self) -> str:
        return titled(f"Lens — {self.lens_input.config.type.value}", self.lens_input.config)

    @property
    def subtitle(self) -> str:
        prompt = self.lens_input.prompt
        read = f'prompt: "{escape(prompt)}"' if prompt is not None else "weights (no prompt)"
        return f"{read} · top-1 of top-{self.lens_input.config.top_k}"

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

    def hover(self, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        lines = [coordinate, describe_row(row, "source_")]
        for i in range(self.lens_input.config.top_k):
            token, probability = f"str_token_{i}", f"probability_{i}"
            if token in row.index and probability in row.index:
                lines.append(f"{i + 1}. {format_value(row[token])} — {format_value(row[probability])}")
        return "<br>".join(line for line in lines if line)
