import abc

import torch

from intex.patching.config import PatchingMetric
from intex.patching.input import PatchingInput


class LogitDiffMixin:
    @classmethod
    def get_logit_diff(
        cls,
        left_logits: torch.Tensor,
        right_logits: torch.Tensor,
        left_generation: int,
        right_generation: int,
    ) -> torch.Tensor:
        return (
            cls.extract_logit(left_logits, left_generation) - cls.extract_logit(right_logits, right_generation)
        ).mean()

    @staticmethod
    def extract_logit(logits: torch.Tensor, token: int) -> torch.Tensor:
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]

        return logits[..., token]


class AbstractMetric(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def compute(patching_input: PatchingInput, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def __call__(cls, patching_input: PatchingInput, logits: torch.Tensor) -> torch.Tensor:
        return cls.compute(patching_input, logits)

    @staticmethod
    def get(metric: PatchingMetric) -> type["AbstractMetric"]:
        match metric:
            case PatchingMetric.LOGIT_DIFF:
                return DiffMetric
            case PatchingMetric.NORMALIZED_LOGIT_DIFF:
                return NormalizedDiffMetric
            case PatchingMetric.KL_DIFF:
                return KLDivMetric
            case _:
                raise ValueError(f"Unknown patching metric: {metric}")


class DiffMetric(AbstractMetric, LogitDiffMixin):
    def compute(self, patching_input: PatchingInput, logits: torch.Tensor) -> torch.Tensor:
        if patching_input.source_generation is not None and patching_input.target_generation is not None:
            return self.get_logit_diff(
                logits, logits, patching_input.source_generation, patching_input.target_generation
            )
        return torch.empty()


class NormalizedDiffMetric(AbstractMetric, LogitDiffMixin):
    def compute(self, patching_input: PatchingInput, logits: torch.Tensor) -> torch.Tensor:
        if patching_input.source_generation is not None and patching_input.target_generation is not None:
            return (
                self.get_logit_diff(logits, logits, patching_input.source_generation, patching_input.target_generation)
                - patching_input.target_baseline
            ) / (patching_input.source_baseline - patching_input.target_baseline)
        return torch.empty()


class KLDivMetric(AbstractMetric, LogitDiffMixin):
    def compute(self, patching_input: PatchingInput, logits: torch.Tensor) -> torch.Tensor:
        source_probabilities = patching_input.source_probabilities
        target_probabilities = patching_input.target_probabilities
        if source_probabilities is not None and target_probabilities is not None:
            return torch.nn.functional.kl_div(
                source_probabilities.log(), target_probabilities, reduction="batchmean"
            ) - torch.nn.functional.kl_div(source_probabilities.log(), logits.softmax(dim=-1), reduction="batchmean")
        return torch.empty()
