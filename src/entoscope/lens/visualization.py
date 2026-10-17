"""Interactive Plotly rendering of a lens run.

The grid is the run's own index - layer down, position (or head, or neuron) across - coloured by a
rank's probability, with that rank's token printed in the cell and the whole top-k in hover. One
grid is drawn per rank, so a run's second and third guesses are as visible as its first.
"""

from __future__ import annotations

from html import escape

from entoscope.lens.input import LensInput
from entoscope.model.token import str_token_labels
from entoscope.operation.visualization import GridFigure, describe_row, format_value, titled


class LensFigure(GridFigure[LensInput, int]):
    @property
    def title(self) -> str:
        return titled(f"Lens - {self.operation_input.config.method.type}", self.operation_input.config)

    def panels(self) -> list[int]:
        return list(range(self.operation_input.config.top_k))

    def heading(self, rank: int) -> str:
        config = self.operation_input.config
        return titled(f"Lens - {config.method.type} · top-{rank + 1}", config)

    def subtitle(self, rank: int) -> str:
        prompt = self.operation_input.prompt
        read = f'prompt: "{escape(prompt)}"' if prompt is not None else "weights (no prompt)"
        return f"{read} · top-{rank + 1} of top-{self.operation_input.config.top_k}"

    def value_column(self, rank: int) -> str:
        return f"probability_{rank}"

    def value_title(self, rank: int) -> str:
        return "probability"

    def bar_yaxis(self, rank: int) -> dict:
        return dict(title=self.value_title(rank), range=[0.0, 1.05])

    def cell_text(self, rank: int) -> list[str]:
        return [format_value(token) for token in self.result[f"str_token_{rank}"]]

    def str_tokens(self) -> dict[str, list[str]]:
        if self.operation_input.prompt is None:
            return {}
        return {"source_position": str_token_labels(self.model, self.operation_input.tokens)}

    def hover(self, rank: int, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        lines = [coordinate, describe_row(row, "source_")]
        for i in range(self.operation_input.config.top_k):
            token, probability = f"str_token_{i}", f"probability_{i}"
            if token in row.index and probability in row.index:
                line = f"{i + 1}. {format_value(row[token])} - {format_value(row[probability])}"
                lines.append(f"<b>{line}</b>" if i == rank else line)
        return "<br>".join(line for line in lines if line)
