import pytest
from transformer_lens import HookedTransformer


@pytest.fixture(scope="session")
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    return model
