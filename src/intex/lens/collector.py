import os
from collections.abc import Sequence

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from intex.lens.input import LensInput
from intex.util.collections import dataframe_to_list, to_dict
from intex.util.data import create_if_not_exists, write_json


def escape_token(token: str) -> str:
    """Escape non-printable characters (e.g. "\\r", "\\n" byte tokens) — raw ones break CSV round-trips."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in token)


class LensCollector:
    def __init__(self, lens_input: LensInput, model: HookedTransformer, save_dir: str | None = None):
        self.lens_input = lens_input
        self.model = model
        self.save_dir = save_dir

        self._cells: dict[tuple[int, ...], dict[str, int | str | float]] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        rows = pd.DataFrame([self._cells[tuple(row)] for row in dataframe_to_list(self.lens_input.index.output_df)])
        return pd.concat([self.lens_input.index.to_df().reset_index(drop=True), rows], axis=1)

    def prepare(self):
        self._cells = {}

    def collect(self, logits: torch.Tensor, index: Sequence[int]):
        assert self._cells is not None
        # Each cell holds the top-k tokens of a single-position slice; elements spanning
        # multiple positions (e.g. a layer-only location) do not fit this per-row layout.
        probabilities, tokens = logits.detach().softmax(dim=-1).squeeze(0).topk(self.lens_input.config.top_k)

        cell: dict[str, int | str | float] = {}
        for i, (token, probability) in enumerate(zip(tokens.tolist(), probabilities.tolist(), strict=True)):
            cell[f"token_{i}"] = token
            cell[f"str_token_{i}"] = escape_token(self.model.to_single_str_token(token))
            cell[f"probability_{i}"] = probability
        self._cells[tuple(index)] = cell

    def save(self):
        if self.save_dir is not None:
            create_if_not_exists(self.save_dir)

            self.result.to_csv(os.path.join(self.save_dir, "result.csv"), index=False)
            self.lens_input.index.to_csv(os.path.join(self.save_dir, "index.csv"))
            write_json(os.path.join(self.save_dir, "config.json"), to_dict(self.lens_input.config))

            print(f"{self.lens_input.config.config_id} done - results saved to {self.save_dir}")
