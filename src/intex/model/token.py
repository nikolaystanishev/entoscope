import torch
from jaxtyping import Int

from intex.model.base import Model


def escape_token(token: str) -> str:
    """Escape non-printable characters (e.g. "\\r", "\\n" byte tokens) — raw ones break CSV round-trips."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in token)


def str_token(model: Model, token: int) -> str:
    """One token's string, CSV/XML-safe."""
    return escape_token(model.to_single_str_token(token))


def str_token_labels(model: Model, tokens: Int[torch.Tensor, "batch pos"]) -> list[str]:
    """Human-readable, CSV/XML-safe token strings for one prompt (batch item 0)."""
    return [escape_token(token) for token in model.to_str_tokens(tokens[0])]
