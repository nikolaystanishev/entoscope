import torch
from jaxtyping import Float

from intex.lens.method.lens import AbstractLensMethod
from intex.model.config import ModelLocatorConfig
from intex.model.index import IndexElement


class LogitLens(AbstractLensMethod):
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        with torch.no_grad():
            return self.model.ln_final(activation) @ self.model.W_U + self.model.b_U
