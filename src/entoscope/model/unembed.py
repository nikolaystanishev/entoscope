import copy

import torch
from tuned_lens.nn.unembed import Unembed

from entoscope.model.base import Model, is_bridge, unwrap
from entoscope.model.weight import ModelWeight, model_weight


class _BridgeUnembed(Unembed):
    """``Unembed`` built from a bridge's own tensors.

    ``Unembed.__init__`` reaches the model through ``tuned_lens.model_surgery``, which dispatches
    on ``HookedTransformer`` and ``PreTrainedModel``; a bridge is neither, and its ``original_model``
    carries the unprocessed HuggingFace weights rather than the ones the bridge reads.
    """

    def __init__(self, model: Model):
        torch.nn.Module.__init__(self)
        unembedding = torch.nn.Linear(model.cfg.d_model, model.cfg.d_vocab_out)
        unembedding.weight.data = model_weight(model, ModelWeight.UNEMBED)
        unembedding.bias.data = model.b_U

        self.final_norm = copy.deepcopy(model.ln_final.original_component)
        self.unembedding = unembedding
        self.requires_grad_(False)


def build_unembed(model: Model) -> Unembed:
    transformer = unwrap(model)
    return _BridgeUnembed(transformer) if is_bridge(transformer) else Unembed(transformer)
