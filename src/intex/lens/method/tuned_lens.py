from dataclasses import dataclass
from functools import cached_property, lru_cache

import torch
from jaxtyping import Float
from tuned_lens.nn import TunedLens as PretrainedTunedLens

from intex.lens.config import LensConfig, LensMethodConfig
from intex.lens.method.logit_lens import _QUERY_KEY_ACTIVATIONS, LogitLens, _layer
from intex.model import Model
from intex.model.config import ModelLayerLocation, ModelLocatorConfig
from intex.model.index import IndexElement
from intex.model.unembed import build_unembed

# Residual states the translator maps as a whole; every other activation is a component written
# into one of them, and a component's share of the translated state carries none of the bias.
_RESIDUAL_STATES = (
    ModelLayerLocation.RESIDUAL_PRE,
    ModelLayerLocation.RESIDUAL_MID,
    ModelLayerLocation.RESIDUAL_POST,
)


def resource_id(model: Model) -> str:
    resource = model.cfg.tokenizer_name or model.cfg.model_name
    if not resource:
        raise ValueError(
            "Cannot tell which tuned lens belongs to this model: it has neither a tokenizer nor a model name, "
            "so name the translators in the tuned lens's 'translators' field"
        )
    return resource


@lru_cache(maxsize=2)
def load_translators(model: Model, resource: str) -> torch.nn.ModuleList:
    """Pretrained translators for ``resource`` - a directory, or a resource id on the hub space.

    Cached so that repeated runs over the same model read the checkpoint once; the model is part of
    the key only to build the unembedding the checkpoint is checked against.
    """
    lens = PretrainedTunedLens.from_unembed_and_pretrained(build_unembed(model), resource)
    if lens.config.d_model != model.cfg.d_model or lens.config.num_hidden_layers != model.cfg.n_layers:
        raise ValueError(
            f"The tuned lens '{resource}' was trained for a model with {lens.config.num_hidden_layers} layers of "
            f"width {lens.config.d_model}, but this model has {model.cfg.n_layers} of width {model.cfg.d_model}"
        )
    return lens.layer_translators.to(device=model.W_U.device, dtype=model.W_U.dtype).requires_grad_(False)


@dataclass(kw_only=True)
class TunedLensConfig(LensMethodConfig):
    """``translators`` picks the pretrained translators: a resource id on the tuned-lens hub space or a
    local directory holding ``config.json`` and ``params.pt``. Unset, the model's own name is used."""

    translators: str | None = None


class TunedLens(LogitLens[TunedLensConfig]):
    """The logit lens with each residual state's trained translator applied before the model's read-out."""

    def __init__(self, model: Model, config: LensConfig):
        super().__init__(model, config)
        # Resolved now, so a model with no name to look its translators up by fails as the lens is built; loaded
        # on first use, so building a lens - which a plan does - reads no checkpoint.
        self.resource = self.method_config.translators or resource_id(model)

    @cached_property
    def translators(self) -> torch.nn.ModuleList:
        return load_translators(self.model, self.resource)

    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        if index.activation() in _QUERY_KEY_ACTIVATIONS:
            raise ValueError(
                f"{type(self).__name__} cannot project {index.activation().value}: its translator maps the residual "
                "stream onto the final residual, while queries and keys are read back through the other side of the "
                "head - use the logit lens for them"
            )
        return super().project(activation, index)

    def _to_final_residual(
        self,
        residual: Float[torch.Tensor, "... d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        translator_index = self._translator_index(index)
        if translator_index is None:
            return residual

        translator = self.translators[translator_index]
        transformed = residual + torch.nn.functional.linear(residual, translator.weight)
        if translator.bias is not None and index.activation() in _RESIDUAL_STATES:
            transformed = transformed + translator.bias
        return transformed

    def _translator_index(self, index: IndexElement[ModelLocatorConfig]) -> int | None:
        """The translator of the residual state the element is, or the one it writes into.

        Translators are indexed by residual state, as in the model's hidden states: state ``l`` is
        ``resid.pre`` of layer ``l``, and everything layer ``l`` writes lands in state ``l + 1``.
        The last state is the one the model unembeds itself, so it needs no translation.
        """
        if index.activation() is ModelLayerLocation.EMBEDDING:
            return 0

        layer = _layer(index)
        translator_index = layer if index.activation() is ModelLayerLocation.RESIDUAL_PRE else layer + 1
        return translator_index if translator_index < self.model.cfg.n_layers else None
