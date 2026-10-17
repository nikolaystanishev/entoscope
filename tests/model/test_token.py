import pytest
import torch

from entoscope.model.token import str_token_labels

PROMPT = "The capital of France is"


def test_the_tokens_of_one_prompt_are_labelled_with_or_without_a_batch_axis(model):
    tokens = model.to_tokens(PROMPT)

    assert str_token_labels(model, tokens) == str_token_labels(model, tokens[0]) == model.to_str_tokens(PROMPT)


def test_the_tokens_of_several_prompts_are_rejected(model):
    tokens = model.to_tokens(PROMPT)

    with pytest.raises(ValueError, match=r"tokens of a single prompt, got a batch of shape \(2, "):
        str_token_labels(model, torch.cat([tokens, tokens]))
