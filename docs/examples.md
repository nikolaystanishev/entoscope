# Examples

Every operation has an experiment script and a YAML file of configs. A script runs one entry of the
YAML file beside it, picked by `--config-id`, so save the two side by side:

```bash
python lens.py                                      # the file's default entry
python lens.py --config-id tuned_lens               # another entry
python patching.py --config-id head_sweep --dry-run # plan it, without running the model
python lens.py --help                               # every option
```

A run writes its results into `runs/<config-id>/`; open the `.html` page there in a browser. Every
script also takes `--model-name` (default `gpt2`), `--output-dir` (default `runs`) and `--config-path`,
and all but `custom_lens.py` take `--cache-dir` to keep the model's activations on disk between runs.
The prompts are options too, and the YAML files explain every entry in their comments.
[Experiment scripts](guides/experiment-scripts.md) shows how a script turns its config into options.

## Lens

```{literalinclude} ../examples/lens.py
:caption: lens.py
:language: python
```

```{literalinclude} ../examples/lens.yaml
:caption: lens.yaml
:language: yaml
```

## Probing

```{literalinclude} ../examples/probe.py
:caption: probe.py
:language: python
```

```{literalinclude} ../examples/probe.yaml
:caption: probe.yaml
:language: yaml
```

## Activation patching

```{literalinclude} ../examples/patching.py
:caption: patching.py
:language: python
```

```{literalinclude} ../examples/patching.yaml
:caption: patching.yaml
:language: yaml
```

## Circuit discovery

```{literalinclude} ../examples/circuit.py
:caption: circuit.py
:language: python
```

```{literalinclude} ../examples/circuit.yaml
:caption: circuit.yaml
:language: yaml
```

## A custom lens

A lens the script registers itself, as [Custom methods](guides/custom-methods.md) describes, and the
configs that name it.

```{literalinclude} ../examples/custom_lens.py
:caption: custom_lens.py
:language: python
```

```{literalinclude} ../examples/custom_lens.yaml
:caption: custom_lens.yaml
:language: yaml
```
