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
Python 3.11 to 3.14 (and on 3.15, allowed to fail until torch publishes wheels for it); a `coverage` job combines the versions into the job summary and a `coverage-html` artifact
(report-only, no threshold), and a `docs` job builds the site above into a `docs-html` artifact. A `lowest`
job runs the tests on the lowest dependency versions `pyproject.toml` allows, and a `package` job builds the
sdist and wheel and runs the tests against the installed wheel. A `v*` tag runs `release.yml`: the whole of
CI, then PyPI by trusted publishing, the docs to GitHub Pages and a GitHub release. CI's jobs run on the
runner the repository variable `CI_RUNNER` names, `ubuntu-latest` when it is unset; a self-hosted one must
not be named once the repository is public, since a fork's pull request runs its own workflow on it.

## What this repo is

entoscope - mechanistic interpretability modular framework for analysis of transformer models, currently supporting
patching, probing, lensing, and circuit discovery.

## Architecture

Every operation - Logit/Tuned/Parameter Lens, Activation Patching, Probing, Circuit Discovery -
follows the same four-part shape, and adding one means writing the same four pieces:

- **Config** (`<op>/config.py`) - a `kw_only` dataclass extending `AbstractConfig`, validating in
  `__post_init__`. Loadable from YAML through `ConfigFactory[TheConfig](path).create("id")`, which
  deserializes strictly via `entoscope.util.data.from_dict` (unknown/missing keys and mistyped scalars
  raise; a `from_config` classmethod on a field's type overrides the default handling). `from_dict` is
  the only reader of raw shapes: a config built in code holds typed values, and code holding a raw
  shape reads it through `from_dict`. Its `method` field is
  the method config, a `MethodConfig` whose `type` names the method - in YAML a mapping or just
  the type. `type` is `MethodConfig`'s own field and has no default, since one method config class
  serves several methods. It is saved as `config.json` beside the run's results,
  which `from_dict` reads back - so a config holds only data, and `to_dict` rejects what has no JSON form.
- **Input** (`<op>/input.py`) - a dataclass extending `OperationInput` that holds prompts and the
  lazily-filled tensors. `prepare(model)` resolves everything that needs no forward pass, the index,
  and then gathers activations (nothing more for an input whose method runs the model itself);
  `prepare(model, index_only=True)`, which `plan()` calls, stops after the index. Each step is guarded
  on what it fills, so `plan()` followed by `apply()` on the same object costs one index resolution.
  What needs no model is checked in `__post_init__`, so a bad config fails as the input is built.
- **Operator** (`<op>/<name>.py`) - `apply(input, save_dir, return_collector, verbose)` iterates
  `input.index`, and `plan(input, save_dir, verbose)` builds an `OperationPlan` describing the same
  run without touching the model or `save_dir`. Constructor takes `(model, collector_class=...)`.
- **Collector** (`<op>/collector.py`) - allocates its result as it is built, accumulates one cell per
  index row and writes the run's files. `FrameCollector` glues cells onto the index's own columns to produce `result.csv`;
  `VisualizationCollector` additionally renders `result.html`.

An operation (or a method) also gets its exports in its package's `__init__.py` (see
[Public API](#public-api)), its page in `docs/operations/`, its modules in `docs/reference/` and an
entry in `examples/` - see [Docs](#docs).

### The index is the spine

`ModelLocatorConfig` (`model/config.py`) is what every operation's config embeds to say *where* in
the model to look, in one of two mutually exclusive modes:

- declarative - `location` names the axes (`layer`, `head`, `neuron`, `position`, and `attn.pattern`'s
  `dest_pos` / `src_pos`), the full range of each is generated, then narrowed by `Filter` and reshaped
  by the `Group` / `Incremental` / `Combination` transforms;
- explicit - `coordinates` enumerates exact points, optionally each with its own `activation`.

`parse(model, n_positions)` turns that into a DataFrame. `n_positions=None` means the prompt length
varies (probing over unequal-length prompts): positions stay symbolic so torch resolves a negative
against each prompt's own cache at slice time, and an unfiltered position axis (`POSITION_AXES`) is
rejected. A known axis size bounds every value (`count_from_end`), and a locator may name only the
axes its activation's tensor has (`located_axes` in `model/slicer.py`), plus `layer` inside a block.

`Index` (`model/index.py`) wraps the DataFrame(s) and is what operators iterate. A source and a target
pair row by row by position: as many rows, or one side of a single row repeated along the other; any
other pairing raises as the index is built. Each `SingleIndex`
carries an `output` position - the row a collector files its cell under - plus a source
`IndexElement` and, for patching, a target one. `IndexElement.slices()` turns a row into the tuple
of tensor slices for that activation, via `ActivationSlicer` (`model/slicer.py`), which owns the
axis order per activation kind (`[batch, pos, neuron]`, `[batch, pos, head, neuron]`,
`[batch, head, dest_pos, src_pos]`). The tuple holds one value per axis and is applied to a tensor
with `IndexElement.read_slices` / `write_slices` (`read_unit_slices` for a weight laid out per unit),
which index every axis with a tensor laid along its own axis of the block, so lists on several axes
read and write every combination of their values; indexing with the tuple directly would pair the
lists. `transformer_activation()` maps entoscope's dotted `ModelLayerLocation` values onto
transformer activation names. pandas 3 stores a frame's enum cells as plain `str`, so a row's
activation is read through `IndexElement.activation()`, which returns the member.

Adding a location or activation therefore means touching `ModelLayerLocation`,
`MODEL_LOCATION_MAPPER`, `TRANSFORMER_ACT_NAME` (`model/activation.py`, names relative to the block)
and `ActivationSlicer` together; a new axis also goes in `AXIS_ORDER` (`operation/visualization.py`),
and in `POSITION_AXES` when the prompt sizes it. The `Index` holds only the rows; the transformer
activations a run touches are its input's `activation_names()`. Which exist depends on the backend and
the architecture, so an input checks those names against the model as its index is resolved
(`check_transformer_activations`). A patch writes in place, which is what reaches a bridge's residual
stream, except into a sublayer's input, which the block may hand on elsewhere too (`writable`); both
live in `model/activation.py`.

### Models and backends

`Model` (`model/base.py`) is a union: `TransformerBridge | HookedTransformer | CachedModel`.
`TransformerBridge` with raw HuggingFace numerics is the default; `enable_compatibility_mode()`
switches to `transformer_lens`'s folded-LayerNorm convention and both must keep working.
`HookedTransformer` is kept because Intel macOS is pinned to `transformer-lens` 2.x (see the
platform markers in `pyproject.toml`), and transformer-lens 4 removed it, so either backend may be
missing. **Any new `transformer_lens` import that moved or went away between 2.x, 3.x and 4.x needs
the same `try/except ImportError` fallback** used in `model/base.py` and `tests/conftest.py`.
Backend-dependent weight access is centralized in `model/weight.py` and `model/unembed.py` - go
through those rather than reading `model.W_*` directly. `model_weight` returns any `ModelWeight` with
`d_model` last, whichever side of the residual stream it sits on, and `ACTIVATION_WEIGHT` maps an
activation to its own weight; the lenses read every weight through them.

`CachedModel` (`model/cache.py`) is a transparent read-through disk cache that overrides only
`run_with_cache` and forwards everything else via `__getattr__`, a method returning the model
returning the wrapper; `cache_dir=None` makes it a pass-through. Cache keys fold in a sampled digest
of the model's weights, taken on every call, so neither a fine-tune nor an edit after wrapping
(compatibility mode, a dtype change) can read another model's entries.

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
`register` users call. The config's `method.type` picks one, and the factory's `create(model, config)` builds
it from `(model, config)`, taking `config.method` as its `self.method_config`. Each op's method base is generic in the
dataclass of its method config: a method's first type argument (`LogitLens[TheConfig]`) types `self.method_config`
and is what `method_config_class` (`operation/method.py`) reads to parse a config, a TypeVar left open (the op
bases and the built-ins without fields of their own) giving its bound. Each op's method config base extends
`MethodConfig` (`operation/method.py`) and names its factory in `factory()` (imported there, as the
factory imports the methods, which import the method config): its `from_config` reads a file's `method`
strictly into the class its `type` names, so a bad name or field fails as the file is read; one built in code
is taken as given. Each `<op>/__init__.py`
imports its factory so the built-ins are registered whatever is imported first. A lens is a single
class that tells activations apart by `index.activation()` itself. `DatasetFactory` is the same registry for
datasets: `DatasetConfig` is a `MethodConfig`, and a dataset's type argument names its config class;
entoscope ships no datasets.

Everything else dispatches by `match` on an enum in an `AbstractX.get(enum)` static method
(`patching/metric.py`'s metrics and comparisons, `patching/processor.py`'s source/target strategies,
`probe/metric.py`, `probe/pool.py`, `model/slicer.py`).

Patching composes three of those: `PatchingMetric` picks the scalar read off a run, `MetricComparison`
picks what the patched run is reported against, and the config's `method` picks how the change in
the metric is obtained. Patched run `i` applies row `i` of every patch element, and the result is laid
out along the element with the most rows. The config's `metric` maps a name of its choosing to a
`PatchingMetricConfig`: the `PatchingMetric`, the tokens it reads by their names in the input's
`tokens`, and its comparison. The input builds the metrics and owns their lifetime. Every method yields
the same primitive - the patched run's metric minus the unpatched target run's, one per *metric* -
so adding one means implementing `AbstractPatchingMethod.scores` (and `passes`, which a plan reports;
`check` refuses an input before any forward pass); `_report` reports that under each metric's comparison
and the figure draws one plot per metric. `integrated_gradients` interpolates one transformer
activation per pass, so its cost grows with the transformer activations an index patches but a row's score never depends on the
rest of the index. A method
that actually runs the patched model (only `exact`) also reports that run's next token as the row's
`generation`; the gradient methods have no such run and leave it out. Backends (MPS notably) round
differently at different batch shapes, so the input's cached source and target runs are made as wide
as the patched runs (`batch_width`) and `exact` pads its last batch to that width: a patch that writes
back what was there must reproduce the target run exactly.

### Public API

Each package's `__init__.py` re-exports its public classes in a sorted `__all__`, and
`entoscope/__init__.py` re-exports the everyday ones from those (operators, inputs, configs, method
configs, enums, `ModelLocatorConfig` and its transforms, `CachedModel`, `ConfigFactory`, `experiment`), so
`from entoscope import Lens, LensConfig, LensInput` works; `tests/test_public_api.py` keeps the levels in
step. The built-in method classes, the index classes and `from_dict`/`to_dict` are
not re-exported: they are imported from the modules defining them. The docs and examples import from these. Code in `src/` imports from the defining module instead,
never from `entoscope` or an operation's package, whose `__init__` imports everything and would cycle.

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
- Docstrings are Google style, rendered by napoleon: a summary line, then `Args:`, `Returns:` (or
  `Yields:`) and `Raises:` for a function, and `Args:` for a class's constructor or `Attributes:` for a
  dataclass's fields. Types stay in the signature, not the docstring.
- Messages to the user go through a module's `logging.getLogger(__name__)`, never `print` or `warnings`.
  `entoscope/__init__.py` gives the `entoscope` logger its own stderr handler at INFO and stops it propagating,
  so a dependency that configures the root logger (transformer_lens does) cannot reformat or mute it.
- `src/entoscope/circuit/vendor/` is CC BY-NC 4.0 third-party code, kept close to verbatim and excluded
  from ruff. Do not reformat or refactor it. Circuit Discovery's `information_flow_routes` method is
  non-commercial-only as a result.
- Line length 120, ruff rules `E,F,UP,B,SIM,I`; `F722` is off for jaxtyping shape strings.

## Docs

`docs/` is a Sphinx site written in Markdown through MyST, with the Furo theme. Each page reads on
GitHub too, so pages stick to plain Markdown, GitHub-style `> [!NOTE]` alerts, relative links and hidden toctrees. The API reference
(`docs/reference/`) is a page per package that lists its modules for autodoc - a new module is added
there by hand - which makes docstrings part of the docs: they must be valid reStructuredText, since
the `-W` build fails on any warning. `README.md` is the landing page and the PyPI description, so its
links are absolute ones into the published site; `examples/` holds an experiment script and a YAML file
of configs per operation, which `docs/examples.md` shows on the site.

The docs are tested with the code. `tests/docs/test_snippets.py` executes every ```` ```python ````
block of `README.md` and `docs/`, page by page in one namespace (a block preceded by
`<!-- docs-test: skip -->` is display-only), and `tests/examples/test_examples.py` loads every example
config and dry-runs every script on each one. Both run on the bridge backend only, against the
session's GPT-2.

## Tests

Integration and e2e, against real (tiny, randomly initialized) transformers built in
`tests/conftest.py`, plus real GPT-2 for the `model` fixture. A test drives an operation through its
public API (a YAML config, `plan()`, `apply()`, the files a run writes) and checks the result against
an expectation computed independently from the model, not against an input's internal state; the
exceptions are `CachedModel` and the public-API contract. Everything derives from the session-scoped
`backend` fixture, so each test runs once per backend the installed transformer-lens has (`bridge`,
`compat` - a bridge after `enable_compatibility_mode()` - and `hooked`; 4.x has no `hooked`, so the
`lowest` CI job's 3.x is where it runs) and a test taking several model fixtures gets them all on the
same one - a test reaching a fixture through `request.getfixturevalue` must also name `backend`. Other
shared helpers: `forbid_model_runs` (asserts a `plan()` never runs the model), `record_runs` (records
every forward pass a run makes: its width and what it cached), `rendered_figure` (parses the plotly
JSON back out of `result.html`), `config_from_yaml` (sources a config the way a run does, through
`ConfigFactory`) and `registries` (undoes the registrations a test makes).
