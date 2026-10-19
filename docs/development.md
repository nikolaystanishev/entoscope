# Development

## Setup

```bash
git clone git@github.com:nikolaystanishev/entoscope.git
cd entoscope
uv sync --group dev --group docs
```

## Tests

```bash
uv run pytest
uv run pytest tests/lens/test_logit_lens.py::test_the_grid_covers_every_layer_and_position -x
uv run pytest --cov --cov-report=html
```

The tests are integration and end-to-end tests against real transformers: they run an operation the
way a user does and check its results against values computed independently from the model. Most use
tiny, randomly initialized models built in `tests/conftest.py`, and `model` is GPT-2, downloaded from
the HuggingFace hub on first use. The tuned-lens tests also download GPT-2's pretrained translators.
Every model fixture derives from the session-scoped `backend` fixture, so each test runs once per
backend the installed transformer-lens has: with a `TransformerBridge`, with one in compatibility mode
(`enable_compatibility_mode()`), and on transformer-lens 3 also with a `HookedTransformer`, which
transformer-lens 4 removed.

`tests/conftest.py` also provides:

| Fixture or helper | Use |
|---|---|
| `forbid_model_runs` | makes any forward pass fail, for asserting a `plan` never runs the model |
| `record_runs` | records every forward pass a run makes, its batch width and what it cached |
| `rendered_figure`, `rendered_figures` | parse the plotly figures back out of a written page |
| `config_from_yaml` | load a config the way a run does, through a YAML file and `ConfigFactory` |
| `registries` | undo the registrations a test makes |

## Lint and format

```bash
uv run ruff check . && uv run ruff format --check .
```

Lines are 120 characters, and ruff checks `E`, `F`, `UP`, `B`, `SIM` and `I`.
`src/entoscope/circuit/vendor/` is third-party code kept close to verbatim and is excluded from ruff:
do not reformat it.

## Docs

The docs are Markdown pages in `docs/`, built by Sphinx with MyST, and the API reference is generated
from the docstrings:

```bash
uv run --group docs sphinx-build -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` to read the result. `-W` turns every warning into an error, which
catches broken cross-references and docstrings that are not valid reStructuredText. Docstrings are
Google style (`Args:`, `Returns:`, `Raises:`, and `Attributes:` for a dataclass's fields), with the
types left to the signature.

The docs are tested along with the code, so they cannot drift from it unnoticed:

- `tests/docs/test_snippets.py` runs every `python` block of `README.md` and `docs/`. A page's blocks
  run in order and share one namespace, in a temporary directory holding a copy of `examples/`.
  `TransformerBridge.boot_transformers("gpt2")` returns the test session's GPT-2, so a block must
  load `gpt2`, and must not change the model (such as calling `enable_compatibility_mode()`) or global
  state (such as logging). A block that should not run, such as a fragment or one using another model,
  is marked by the HTML comment `<!-- docs-test: skip -->` on the line before it.
- `tests/examples/test_examples.py` loads every entry of every YAML file in `examples/`, and runs each
  example script on each of its entries with `--dry-run`.

When you add an operation or a method, add or update its docs page, its reference page in
`docs/reference/` and its example.

## CI

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`:

- ruff's lint and format checks;
- `pytest --cov` on Python 3.11 to 3.14, with a job that combines the coverage of all versions into
  the job summary and a `coverage-html` artifact, and on Python 3.15, which may fail until torch
  publishes wheels for it;
- `lowest`, the tests on Python 3.11 with the lowest version of each dependency `pyproject.toml`
  allows, so the lower bounds published with the package are ones the tests pass on. `uv.lock` holds
  transformer-lens 4, which has no `HookedTransformer`, so this job's transformer-lens 3 is also where
  that backend and the bridge parity tests run;
- `package`, which builds the sdist and the wheel, checks their metadata as PyPI will, and runs the
  tests against the installed wheel rather than `src/`, uploading both as a `dist` artifact;
- the docs build, uploaded as a `docs-html` artifact you can download and open.

## Releasing

A release is a `v` tag on `main`, and `.github/workflows/release.yml` does the rest:

1. Bump the version in a pull request, which updates `pyproject.toml` and `uv.lock`:

   ```bash
   uv version --bump patch  # or minor, or major
   ```

2. Once it is merged, tag the merge commit with the new version and push the tag:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

The workflow checks that the tag names the version in `pyproject.toml`, runs the whole of CI on the
tag, and then publishes the sdist and wheel the `package` job built and tested to PyPI. It also deploys
the docs to GitHub Pages and creates a GitHub release whose notes list the pull requests merged since
the last one. PyPI is reached by trusted publishing, so there is no token to keep: the PyPI project
trusts `release.yml` running in the repository's `pypi` environment.

## Architecture

`AGENTS.md` at the repository root describes how the code is put together: the four parts every
operation shares, the index, the model backends, the method registries, and the conventions for
comments, logging and tests.
