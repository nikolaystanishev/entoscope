import torch
from jaxtyping import Int

from entoscope.model.base import Model


def escape_token(token: str) -> str:
    """Escape a token's non-printable characters, since raw ones such as ``\\r`` break CSV round-trips.

    Args:
        token: A token string.

    Returns:
        The token with every non-printable character written as its escape sequence.
    """
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in token)


def str_token(model: Model, token: int) -> str:
    """Spell one token, safe to write into a CSV or XML file.

    Safe to write, not to read back blindly: pandas' default ``read_csv`` takes real tokens such as
    "null", "None" or "NA" for missing values. Read string columns with ``keep_default_na=False`` to
    keep them verbatim, or go by the token id every collector writes beside its string.

    Args:
        model: The model whose tokenizer spells the token.
        token: A token id.

    Returns:
        The token's string, escaped as ``escape_token`` does.
    """
    return escape_token(model.to_single_str_token(token))


def str_token_labels(model: Model, tokens: Int[torch.Tensor, "... pos"]) -> list[str]:
    """Spell the tokens of one prompt, safe to write into a CSV or XML file.

    Args:
        model: The model whose tokenizer spells the tokens.
        tokens: The prompt's tokens, shaped ``[pos]`` or ``[1, pos]``.

    Returns:
        One escaped string per position.

    Raises:
        ValueError: If ``tokens`` holds more than one prompt.
    """
    if tokens.shape[:-1].numel() != 1:
        raise ValueError(f"Expected the tokens of a single prompt, got a batch of shape {tuple(tokens.shape)}")
    return [escape_token(token) for token in model.to_str_tokens(tokens.flatten())]
