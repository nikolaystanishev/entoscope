# Experiment scripts

`@experiment` turns a function `main(config: Config)` into a command-line script. The fields of the
`Config` dataclass become `--kebab-case` options, with their types, defaults and required-ness taken
from the fields themselves.

## A script

```python
import dataclasses
from dataclasses import dataclass, field

from transformer_lens.model_bridge import TransformerBridge

from entoscope.lens.config import LensConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.operation.config import ConfigFactory
from entoscope.script import experiment


@dataclass
class Config:
    config_path: str = field(metadata={"help": "a YAML file of lens configs"})
    config_id: str = "logit_lens"
    prompts: list[str] = field(default_factory=lambda: ["The capital of France is"])
    top_k: int | None = field(default=None, metadata={"help": "override the config's top_k"})
    model_name: str = field(default="gpt2", metadata={"choices": ["gpt2", "gpt2-medium"]})
    dry_run: bool = False


@experiment
def main(config: Config):
    """Run one lens config on some prompts."""
    model = TransformerBridge.boot_transformers(config.model_name)
    model.eval()
    lens_config = ConfigFactory[LensConfig](config.config_path).create(config.config_id)
    if config.top_k is not None:
        lens_config = dataclasses.replace(lens_config, top_k=config.top_k)

    lens = Lens(model)
    for i, prompt in enumerate(config.prompts):
        lens_input = LensInput(lens_config, prompt)
        save_dir = f"runs/{config.config_id}/{i}"
        if config.dry_run:
            lens.plan(lens_input, save_dir)
        else:
            lens.apply(lens_input, save_dir)
```

Saved as `lens_script.py`, it runs as:

```bash
uv run python lens_script.py --config-path examples/lens.yaml --config-id tuned_lens --dry-run
uv run python lens_script.py --config-path examples/lens.yaml --prompts "The Eiffel Tower is in" "Rome is the capital of"
```

and describes itself:

```text
$ uv run python lens_script.py --help
usage: lens_script.py [-h] --config-path CONFIG_PATH [--config-id CONFIG_ID]
                      [--prompts [PROMPTS ...]] [--top-k TOP_K] [--model-name {gpt2,gpt2-medium}]
                      [--dry-run | --no-dry-run]

Run one lens config on some prompts.

options:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        a YAML file of lens configs
  --config-id CONFIG_ID
                        (default: logit_lens)
  --prompts [PROMPTS ...]
                        (default: ['The capital of France is'])
  --top-k TOP_K         override the config's top_k (default: None)
  --model-name {gpt2,gpt2-medium}
                        (default: gpt2)
  --dry-run, --no-dry-run
                        (default: False)
```

## From fields to options

| Field type | Option |
|---|---|
| `str`, `int`, `float`, ... | `--name VALUE`, converted with the type |
| `bool` | `--name` and `--no-name` |
| `X \| None` | as `X`, with `None` when the option is not given |
| `list[X]` | `--name A B C`, any number of values, each converted with `X` |
| `dict[K, V]` | `--name k1:v1 k2:v2`, each split at its first colon and converted with `K` and `V` |
| `tuple[X, Y]` | exactly that many values, each converted with its own type |
| `tuple[X, ...]` | any number of values, converted with `X` |

A field with a default is optional, and its help ends with `(default: ...)`. Put `%(default)s` in the
help text to place the default yourself. A field without a default is a required option. Anything
argparse-specific goes in the field's `metadata`, which is passed to `add_argument` and overrides
what the field implies: `help`, `choices`, `nargs`, `metavar`, a custom `type`, and so on.

The script's description is the decorator's `description=` argument if given
(`@experiment(description="...")`), else a module-level `DESCRIPTION`, else `main`'s docstring.

## When `main` runs

Run as a script, the module is `__main__`, and the decorator parses the command line and calls `main`
as it decorates it. Everything `main` relies on must therefore be defined above it in the file: the
config class, and any method the script registers (see [Custom methods](custom-methods.md)). Imported
from another module, nothing runs.

The decorated function can also be called directly. `main(Config(...))` uses that config and skips
parsing, and `main(argv=["--config-path", "configs.yaml"])` parses the given list instead of the
command line. A `main` that takes no parameters gets no parser and simply runs.

A command line argparse cannot read, such as a missing required option or a value that does not
convert, exits with status 2 and a usage message.

The scripts on the [Examples](../examples.md) page are all built this way.
