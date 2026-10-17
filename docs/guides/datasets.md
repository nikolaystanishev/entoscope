# Datasets

Datasets are an extension point for declaring where prompts come from, with the same shape as an
operation's methods: a config whose `type` names an implementation in a registry, and a dataclass
extending `DatasetConfig` for the settings that implementation takes. A dataset yields plain `dict`
rows. entoscope ships no implementation: you register your own and `DatasetFactory` resolves it by
`type`.

```python
import json
from dataclasses import dataclass
from pathlib import Path

from entoscope.dataset import AbstractDataset, DatasetConfig, DatasetFactory, Row


@dataclass(kw_only=True)
class JsonlDatasetConfig(DatasetConfig):
    path: str
    limit: int | None = None


@DatasetFactory.register("jsonl")
class JsonlDataset(AbstractDataset[JsonlDatasetConfig]):
    def load(self) -> list[Row]:
        lines = Path(self.config.path).read_text().splitlines()
        return [json.loads(line) for line in lines][: self.config.limit]


Path("prompts.jsonl").write_text('{"prompt": "The capital of France is"}\n{"prompt": "The Eiffel Tower is in"}\n')

dataset = DatasetFactory.create(JsonlDatasetConfig(type="jsonl", path="prompts.jsonl", limit=10)).prepare()
print(len(dataset), dataset[0])  # 2 {'prompt': 'The capital of France is'}
```

In a config file:

```yaml
my_data:
  type: jsonl
  path: ${DATA_DIR}/prompts.jsonl
  limit: 10
```

<!-- docs-test: skip -->
```python
from entoscope import ConfigFactory

config = ConfigFactory[DatasetConfig]("configs.yaml").create("my_data")  # a JsonlDatasetConfig
```

A dataset's type argument names its config class, as a method's does: an entry is read strictly into
the class the dataset its `type` names declares, so a missing setting or a key the class does not have
fails as the file is read, and `config.json` reads back into the same class. A dataset with no
settings of its own takes `DatasetConfig` itself, and its entry may be just its `type`.

`load` is called once, lazily, by `prepare()`; `len`, iteration and indexing then read the cached
rows. Operations do not consume datasets yet: `LensInput`, `ProbeInput`, `PatchingInput` and
`CircuitInput` take their prompts directly.

Registering a dataset follows the same rules as registering a method; see
[Registration](custom-methods.md#registration).
