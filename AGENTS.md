# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Commands

```bash
uv sync --group dev
```

```bash
uv run pytest
```

```bash
uv run pytest tests/lens/test_logit_lens.py::test_the_grid_covers_every_layer_and_position -x
```

```bash
uv run ruff check . && uv run ruff format --check .
```

```bash
uv run pytest --cov --cov-report=html
```

```bash
uv run --group docs sphinx-build -W --keep-going docs docs/_build/html
```

CI (`.github/workflows/ci.yml`) runs exactly the lint, format-check and `pytest` above, the tests with `--cov` on
Python 3.10–3.14; a `coverage` job combines the versions into the job summary and a `coverage-html` artifact
(report-only, no threshold), and a `docs` job builds the site above into a `docs-html` artifact.

## What this repo is

entoscope - mechanistic interpretability modular framework for analysis of transformer models, currently supporting
patching, probing, lensing, and circuit discovery.

## Architecture

Every operation - Logit/Tuned/Parameter Lens, Probing, Activation Patching, Circuit Discovery -
follows the same four-part shape, and adding one means writing the same four pieces:

- **Config** (`<op>/config.py`) - a `kw_only` dataclass extending `AbstractConfig`, validating in
  `__post_init__`. Loadable from YAML through `ConfigFactory[TheConfig](path).create("id")`, which
  deserializes strictly via `entoscope.util.data.from_dict` (unknown/missing keys raise; a
  `from_config` classmethod on a field's type overrides the default handling). Its `method` field is
  the method config, a `MethodConfig` whose `type` names the method - in YAML a mapping or just
  the type. `type` is `MethodConfig`'s own field and has no default, since one method config class
  serves several methods. It is saved as `config.json` beside the run's results,
  which `from_dict` reads back - so a config holds only data, and `to_dict` rejects what has no JSON form.
- **Input** (`<op>/input.py`) - a dataclass extending `OperationInput` that holds prompts and the
  lazily-filled tensors. `prepare_index(model)` resolves everything that needs no forward pass;
  `prepare(model)` calls it and then gathers activations. Both are idempotent, guarded on the
  *cache* rather than the index, so `plan()` followed by `apply()` on the same object costs one
  index resolution.
- **Operator** (`<op>/<name>.py`) - `apply(input, save_dir, return_collector, verbose)` iterates
  `input.index`, and `plan(input, save_dir, verbose)` builds an `OperationPlan` describing the same
  run without touching the model or `save_dir`. Constructor takes `(model, collector_class=...)`.
- **Collector** (`<op>/collector.py`) - accumulates one cell per index row and writes the run's
  files. `FrameCollector` glues cells onto the index's own columns to produce `result.csv`;
  `VisualizationCollector` additionally renders `result.html`.

An operation (or a method) also gets its page in `docs/operations/`, its modules in `docs/reference/`
and an entry in `examples/` - see [Docs](#docs).

### The index is the spine

`ModelLocatorConfig` (`model/config.py`) is what every operation's config embeds to say *where* in
the model to look, in one of two mutually exclusive modes:

- declarative - `location` names the axes (`layer`, `head`, `neuron`, `position`), the full range of
  each is generated, then narrowed by `Filter` and reshaped by the `Group` / `Incremental` /
  `Combination` transforms;
- explicit - `coordinates` enumerates exact points, optionally each with its own `activation`.

`parse(model, n_positions)` turns that into a DataFrame. `n_positions=None` means the prompt length
varies (probing over unequal-length prompts): positions stay symbolic so torch resolves a negative
against each prompt's own cache at slice time, and an unfiltered `position` axis is rejected.

`Index` (`model/index.py`) wraps the DataFrame(s) and is what operators iterate. Each `SingleIndex`
carries an `output` position - the row a collector files its cell under - plus a source
`IndexElement` and, for patching, a target one. `IndexElement.slices()` turns a row into the tuple
of tensor slices for that activation, via `ActivationSlicer` (`model/slicer.py`), which owns the
axis order per activation kind (`[batch, pos, neuron]`, `[batch, pos, head, neuron]`,
`[batch, head, dest_pos, src_pos]`). `transformer_activation()` maps entoscope's dotted
`ModelLayerLocation` values onto `transformer_lens` hook names.

Adding a location or activation therefore means touching `ModelLayerLocation`,
`MODEL_LOCATION_MAPPER`, `TRANSFORMER_ACT_NAME` and `ActivationSlicer` together.

### Models and backends

`Model` (`model/base.py`) is a union: `TransformerBridge | HookedTransformer | CachedModel`.
`TransformerBridge` with raw HuggingFace numerics is the default; `enable_compatibility_mode()`
switches to `transformer_lens`'s folded-LayerNorm convention and both must keep working.
`HookedTransformer` is kept because Intel macOS is pinned to `transformer-lens` 2.x (see the
platform markers in `pyproject.toml`). **Any new `transformer_lens` import that moved between 2.x
and 3.x needs the same `try/except ImportError` fallback** used in `model/index.py`,
`model/base.py` and `tests/conftest.py`. Backend-dependent weight access is centralized in
`model/weight.py` and `model/unembed.py` - go through those rather than reading `model.W_*`
directly.

`CachedModel` (`model/cache.py`) is a transparent read-through disk cache that overrides only
`run_with_cache` and forwards everything else via `__getattr__`; `cache_dir=None` makes it a
pass-through. Cache keys fold in a sampled digest of the model's weights, so a fine-tune or a
supplied `hf_model` cannot collide with the base model.

### Visualization is a collector choice, not a config flag

Passing `collector_class=LensCollector` / `PatchingCollector` / `CircuitCollector` opts out of HTML;
each HTML collector lives beside the base one it extends, in `<op>/collector.py`. A `Figure` is a
page of *panels* - `panels()` names them and `figure(panel)` draws one plot each, which is how a
patching page gets a plot per metric column, a lens page one per top-k rank and a circuit page its
routes and its full graph.
`GridFigure` (`operation/visualization.py`) derives the whole layout from the `Index`: position on x
and layer on y when both vary, otherwise `AXIS_ORDER` with the last varying axis across; a
one-dimensional index becomes a bar chart. That layout is shared by every panel, so a subclass
supplies only the per-panel payload (`value_column`, `hover`, `value_range`, ...).

### Dispatch

An operation's methods are an open registry: each `<op>/method/factory.py` is a `RegistryFactory`
(`operation/registry.py`), which knows its methods only by name, the built-ins registered through the same
`register` users call. The config's `method.type` picks one, and a method is built from
`(model, config)`, taking `config.method` as its `self.method_config`. Each op's method base is generic in the
dataclass of its method config: a method's first type argument (`LogitLens[TheConfig]`) types `self.method_config`
and is what `method_config_class` (`operation/method.py`) reads to parse a config, a TypeVar left open (the op
bases and the built-ins without fields of their own) giving its bound. Each op's method config base extends
`MethodConfig` (`operation/method.py`) and names its factory in `factory()` (imported there, as the
factory imports the methods, which import the method config): its `from_config` reads a file's `method`
strictly into the class its `type` names, so a bad name or field fails as the file is read; one built in code
is taken as given. Each `<op>/__init__.py`
imports its factory so the built-ins are registered whatever is imported first. A lens is a single
class that tells activations apart by `index.activation()` itself. `DatasetFactory` is the same plain name registry; entoscope ships no datasets.

Everything else dispatches by `match` on an enum in an `AbstractX.get(enum)` static method
(`patching/metric.py`'s metrics and comparisons, `patching/processor.py`'s source/target strategies,
`probe/metric.py`, `probe/pool.py`, `model/slicer.py`).

Patching composes three of those: `PatchingMetric` picks the scalar read off a run, `MetricComparison`
picks what the patched run is reported against, and the config's `method` picks how the change in
the metric is obtained. The config's `metric` maps the first onto a list of the second, one result column
per pair, so the comparison is chosen per metric and a metric may carry several. Every method yields
the same primitive - the patched run's metric minus the unpatched target run's, one per *metric* -
so adding one means implementing `AbstractPatchingMethod.scores` and nothing else; `_report` fans
that out into the columns and the figure draws one plot per `config.metric_pairs()` entry. A method
that actually runs the patched model (only `exact`) also reports that run's next token as the row's
`generation`; the gradient methods have no such run and leave it out. Backends (MPS notably) round
differently at different batch shapes, so the input's cached source and target runs are made as wide
as the patched runs (`batch_width`) and `exact` pads its last batch to that width: a patch that writes
back what was there must reproduce the target run exactly.

### Scripts

`@entoscope.script.experiment.experiment` turns `main(config: SomeDataclass)` into a CLI: fields become
`--kebab-case` options with types, defaults and required-ness derived from the dataclass
(`script/argument.py` handles `list`/`dict`/`tuple`/`bool`/`Optional`; anything argparse-specific
goes in `field(metadata={...})`). When the decorated function is in `__main__` it runs at decoration
time, so the config class must be defined above it.

## Conventions

- Comments and docstrings explain *why* and what is non-obvious (a numerical gotcha, why a guard
  exists, what a backend does differently). They do not restate what the code or the math already
  says; such comments get deleted.
- Messages to the user go through a module's `logging.getLogger(__name__)`, never `print`.
  `entoscope/__init__.py` gives the `entoscope` logger its own stderr handler at INFO and stops it propagating,
  so a dependency that configures the root logger (transformer_lens does) cannot reformat or mute it.
- `src/entoscope/circuit/vendor/` is CC BY-NC 4.0 third-party code, kept close to verbatim and excluded
  from ruff. Do not reformat or refactor it. Circuit Discovery's `information_flow_routes` method is
  non-commercial-only as a result.
- Line length 120, ruff rules `E,F,UP,B,SIM,I`; `F722` is off for jaxtyping shape strings.

## Docs

`docs/` is a Sphinx site written in Markdown through MyST, with the Furo theme. Each page reads on
GitHub too, so pages stick to plain Markdown, relative links and hidden toctrees. The API reference
(`docs/reference/`) is a page per package that lists its modules for autodoc - a new module is added
there by hand - which makes docstrings part of the docs: they must be valid reStructuredText, since
the `-W` build fails on any warning. `README.md` is the landing page
and points into `docs/`; `examples/` holds an experiment script and a YAML file of configs per
operation.

The docs are tested with the code. `tests/docs/test_snippets.py` executes every ```` ```python ````
block of `README.md` and `docs/`, page by page in one namespace (a block preceded by
`<!-- docs-test: skip -->` is display-only), and `tests/examples/test_examples.py` loads every example
config and dry-runs every script on each one. Both run on the bridge backend only, against the
session's GPT-2.

## Tests

Integration-style, against real (tiny, randomly initialized) transformers built in
`tests/conftest.py`, plus real GPT-2 for the `model` fixture. Everything derives from the
session-scoped `backend` fixture, so each test runs once per backend (`bridge`, `hooked`) and a test
taking several model fixtures gets them all on the same one - a test reaching a fixture through
`request.getfixturevalue` must also name `backend`. Other shared helpers: `forbid_model_runs` (asserts
a `plan()` never runs the model), `rendered_figure` (parses the plotly JSON back out of
`result.html`), `config_from_yaml` (sources a config the way a run does, through `ConfigFactory`) and
`registries` (undoes the registrations a test makes).
