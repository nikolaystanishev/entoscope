from __future__ import annotations

import pandas as pd
import torch

from intex.model.config import ModelLocation, ModelLocationIndex, make_df_from_ranges
from intex.model.index import IndexElement
from intex.operation.index import Index
from intex.patching.config import SinglePatchingConfig
from intex.util.collections import dataframe_to_list, zip_fill


class SinglePatchingIndex:
    def __init__(
        self,
        config: SinglePatchingConfig,
        source_index_names: list[str],
        target_index_names: list[str],
        source_index: list[int],
        target_index: list[int],
        output_index: list[int],
    ):
        self.source = IndexElement(config.source, source_index_names, source_index)
        self.target = IndexElement(config.target, target_index_names, target_index)
        self.output = tuple(output_index)

    def set_batch_index(self, batch_index: int) -> SinglePatchingIndex:
        self.source.set_batch_index(batch_index)
        self.target.set_batch_index(batch_index)

        return self


class PatchingIndex(Index):
    def __init__(self, config: SinglePatchingConfig, source: ModelLocationIndex, target: ModelLocationIndex):
        self.config = config
        self.source = source
        self.target = target

        output_df = self.target.df if len(self.target.df) >= len(self.source.df) else self.source.df
        if self.config.source.coordinates is not None and self.config.target.coordinates is not None:
            self.axis_range = [len(output_df)]
            self.output_df = make_df_from_ranges(["coordinate"], self.axis_range)
        else:
            # Keep one output row per index row (the df may be a sparse, correlated set rather than
            # a full grid), mapping each column's values to dense first-appearance codes.
            coordinate_df = output_df[[col for col in output_df.columns if col != ModelLocation.ACTIVATION]]
            factorized = {col: pd.factorize(coordinate_df[col].astype(str)) for col in coordinate_df.columns}
            self.axis_range = [len(uniques) for _, uniques in factorized.values()]
            self.output_df = pd.DataFrame(
                {col: codes for col, (codes, _) in factorized.items()}, index=coordinate_df.index
            )

    def prepare_output_placeholder(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.axis_range, device=device)

    def to_csv(self, path: str):
        pd.concat(
            [
                self.source.df.add_prefix("source_"),
                self.target.df.add_prefix("target_"),
                self.output_df.add_prefix("index_"),
            ],
            axis=1,
        ).to_csv(path, index=False)

    def __iter__(self):
        return map(
            lambda el: SinglePatchingIndex(
                self.config, self.source.df.columns.to_list(), self.target.df.columns.to_list(), *el
            ),
            zip_fill(
                dataframe_to_list(self.source.df), dataframe_to_list(self.target.df), dataframe_to_list(self.output_df)
            ),
        )

    def __len__(self) -> int:
        return len(self.output_df)
