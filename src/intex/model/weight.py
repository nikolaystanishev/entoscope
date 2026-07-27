import einops
import torch

from intex.model.base import Model, is_bridge

# The bridge submodule each HookedTransformer-style MLP weight is an alias for.
_MLP_SUBMODULE = {"W_in": "in", "W_gate": "gate", "W_out": "out"}


def mlp_weight(model: Model, layer: int, attribute: str) -> torch.Tensor | None:
    """A block's MLP weight in HookedTransformer's orientation.

    ``W_in`` and ``W_gate`` come back as ``[d_model, d_mlp]`` and ``W_out`` as ``[d_mlp, d_model]``.
    A bridge aliases these straight onto the wrapped module's parameter, and the two layouts
    disagree: ``nn.Linear`` stores ``[out, in]`` while GPT-2's ``Conv1D`` stores ``[in, out]``, so
    only the former needs transposing. ``enable_compatibility_mode`` does not reconcile them.
    """
    mlp = model.blocks[layer].mlp
    weight = getattr(mlp, attribute)
    if weight is None or not is_bridge(model):
        return weight

    component = mlp._modules[_MLP_SUBMODULE[attribute]].original_component
    return weight.T if isinstance(component, torch.nn.Linear) else weight


def attention_out_weight(model: Model, layer: int) -> torch.Tensor:
    """A block's ``W_O`` in HookedTransformer's orientation, ``[head, d_head, d_model]``.

    A bridge picks its reshape by reading the layout off the matrix shape, and ``W_O`` is square
    whenever ``d_model == n_heads * d_head`` — so on the architectures ``mlp_weight`` describes it
    takes the ``Conv1D`` branch and skips the transpose ``nn.Linear`` needs. Rebuilding from the
    parameter also covers the non-square case, where the bridge's reading is already right.
    """
    attn = model.blocks[layer].attn
    out = attn._modules.get("o") if is_bridge(model) else None
    if out is None or not isinstance(out.original_component, torch.nn.Linear):
        return attn.W_O
    return einops.rearrange(out.weight.T, "(head d_head) d_model -> head d_head d_model", head=model.cfg.n_heads)
