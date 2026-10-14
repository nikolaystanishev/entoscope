# Configuration files

A config is a dataclass (`LensConfig`, `ProbeConfig`, `PatchingConfig`, `CircuitConfig`). You can
build one in code, or read it from a YAML file and get the same object back. A file is the usual way to
keep the configs of an experiment together and to run them from a script.

## A config file

A file holds any number of configs, each under the id you load it by:

```yaml
logit_lens:
  method: logit_lens
  top_k: 5
  source:
    activation: resid.post
    location: [layer, position]

tuned_lens:
  method: tuned_lens
  top_k: 5
  source:
    activation: resid.post
    location: [layer, position]
```

`ConfigFactory` reads one entry into the config class it is subscripted with:

```python
from entoscope import ConfigFactory, LensConfig

config = ConfigFactory[LensConfig]("examples/lens.yaml").create("logit_lens")
print(config.config_id)  # logit_lens
```

The type argument is required, because it is the class the entry is read into. The entry's id becomes
the config's `config_id`, which titles the run's plan and result pages. A factory reads its file once,
so creating several entries from one factory reads it a single time.

The keys are the config's field names, and the values are their plain forms. Enums are written by
value (`resid.post`, `logit_diff`, `mean`), locator keys as strings (`layer`, `head`), and filters and
coordinates as the lists and mappings shown in [Locating activations](locating.md). Each operation
page lists its config's fields.

## Reading is strict

An entry is checked as it is read, before any model is involved. A key the config does not have, a
required key left out, or an enum value that does not exist raises a `ValueError` that names the
class and the key:

```text
ValueError: Unknown keys for LensConfig: ['top-k']
```

The checks go into `method` too: a method nothing registered raises and lists the methods that are
registered, and a field the method does not take is an unknown key of its method config.

## The `method` field

`method` picks the operation's method and holds its settings. In a file it is either the method's
name alone, which uses its defaults, or a mapping with `type` and the fields to set:

```yaml
method: tuned_lens
```

```yaml
method:
  type: tuned_lens
  translators: gpt2
```

In code it is an instance of the method's config class, which always names its `type`. Methods with
no settings of their own take their operation's base class:

```python
from entoscope import LensMethodConfig, TunedLensConfig

logit = LensMethodConfig(type="logit_lens")
tuned = TunedLensConfig(type="tuned_lens", translators="gpt2")
```

`dataclasses.replace(config, method=tuned)` switches a built config to another method. The operation
pages list every built-in method's fields, and [Custom methods](../guides/custom-methods.md) shows how
a method names its config class.

## Environment variables

A value may name environment variables as `${NAME}`. Each is replaced by the variable's value, or by
`NAME` itself when the variable is unset. A value that is a single variable holding only digits
becomes an int:

```yaml
my_data:
  type: jsonl
  path: ${DATA_DIR}/prompts.jsonl
  limit: ${LIMIT}
```

## Reusing parts of a file

YAML anchors and aliases share a block between entries, which keeps the two sides of a patch, or a
sweep repeated with several methods, written once:

```yaml
residual_sweep:
  metric:
    logit_diff: [normalized]
  method: exact
  patch:
    - source: &residual
        activation: resid.pre
        location: [layer, position]
        type: replace
      target: *residual

attribution:
  metric:
    logit_diff: [indirect_effect]
  method: attribution
  patch:
    - source: *residual
      target: *residual
```

## The config a run leaves behind

Every run writes the config it was made under to `config.json` beside its results, with every field,
defaults included. `from_dict` reads it back into the same config:

```python
import json
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from entoscope import Lens, LensInput, from_dict

model = TransformerBridge.boot_transformers("gpt2")
model.eval()
Lens(model).apply(LensInput(config, "The capital of France is"), save_dir="runs/logit_lens", verbose=False)

saved = from_dict(LensConfig, json.loads(Path("runs/logit_lens/config.json").read_text()))
print(saved == config)  # True
```

A config therefore holds only data: numbers, strings, enums and paths, and lists, tuples, sets,
mappings and dataclasses of them. A field with no such form, such as a function or a tensor, cannot be
written to `config.json`, and saving the run raises a `TypeError`.
