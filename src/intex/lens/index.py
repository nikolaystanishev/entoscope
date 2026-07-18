from __future__ import annotations

import pandas as pd
import torch

from intex.model.config import ModelLocation, ModelLocationIndex, ModelLocatorConfig, make_df_from_ranges
from intex.model.index import IndexElement
from intex.operation.index import Index
from intex.util.collections import dataframe_to_list


class SingleLensIndex:
    def __init__(
        self,
        config: ModelLocatorConfig,
        index_names: list[str],
        index: list[int | list[int] | str],
        output_index: list[int],
    ):
        self.source = IndexElement(config, index_names, index)
        self.output = tuple(output_index)

    def set_batch_index(self, batch_index: int) -> SingleLensIndex:
        self.source.set_batch_index(batch_index)

        return self


class LensIndex(Index):
    def __init__(self, config: ModelLocatorConfig, source: ModelLocationIndex):
        self.config = config
        self.source = source

        if self.config.coordinates is not None:
            self.axis_range = [len(self.source.df)]
            self.output_df = make_df_from_ranges(["coordinate"], self.axis_range)
        else:
            # Keep one output row per index row (the df may be a sparse, correlated set rather than
            # a full grid), mapping each column's values to dense first-appearance codes.
            coordinate_df = self.source.df[[col for col in self.source.df.columns if col != ModelLocation.ACTIVATION]]
            factorized = {col: pd.factorize(coordinate_df[col].astype(str)) for col in coordinate_df.columns}
            self.axis_range = [len(uniques) for _, uniques in factorized.values()]
            self.output_df = pd.DataFrame(
                {col: codes for col, (codes, _) in factorized.items()}, index=coordinate_df.index
            )

    def prepare_output_placeholder(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.axis_range, device=device)

    def to_csv(self, path: str):
        pd.concat([self.source.df.add_prefix("source_"), self.output_df.add_prefix("index_")], axis=1).to_csv(
            path, index=False
        )

    def __iter__(self):
        return map(
            lambda el: SingleLensIndex(self.config, self.source.df.columns.to_list(), *el),
            zip(dataframe_to_list(self.source.df), dataframe_to_list(self.output_df), strict=True),
        )

    def __len__(self) -> int:
        return len(self.output_df)
