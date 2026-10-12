"""The documentation's code, run: every python block of a page, in order, in one namespace per page.

A page reads like a session - a block may use what an earlier block on the page defined - so it is run
as one. A block preceded by ``<!-- docs-test: skip -->`` is shown, not run: a fragment, or code that
would change the shared model or global state.
"""

import re
import shutil
from pathlib import Path

import pytest

from ..conftest import TransformerBridge

ROOT = Path(__file__).parents[2]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
SKIP = "<!-- docs-test: skip -->"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def python_blocks(page: Path) -> list[tuple[int, str]]:
    """The blocks of ``page`` that run, each with the line its code starts on."""
    text = page.read_text(encoding="utf-8")
    blocks = []
    for match in PYTHON_BLOCK.finditer(text):
        line_before = text[: match.start()].rstrip("\n").rpartition("\n")[2]
        if line_before.strip() != SKIP:
            blocks.append((text.count("\n", 0, match.start()) + 2, match.group(1)))
    return blocks


def _boot_session_model(model):
    def boot_transformers(model_name, *args, **kwargs):
        # The docs load GPT-2, which the session already holds; another model would be a download.
        assert model_name == "gpt2", f"a docs block loads {model_name!r}; load 'gpt2' or mark the block skipped"
        return model

    return staticmethod(boot_transformers)


@pytest.mark.parametrize(
    "page", [page for page in PAGES if python_blocks(page)], ids=lambda page: str(page.relative_to(ROOT))
)
def test_the_page_runs(page, backend, request, registries, monkeypatch, tmp_path):
    if backend != "bridge":
        pytest.skip("the docs load their models as a TransformerBridge")
    monkeypatch.setattr(TransformerBridge, "boot_transformers", _boot_session_model(request.getfixturevalue("model")))
    # The pages read the example configs by their path in the repository and write runs/ beside them.
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    monkeypatch.chdir(tmp_path)

    namespace = {"__name__": "docs_snippet"}
    for line, code in python_blocks(page):
        # Padded so a traceback points at the page's own line numbers.
        exec(compile("\n" * (line - 1) + code, str(page), "exec"), namespace)
