# Datasets

Datasets are an extension point for declaring where prompts come from, with the same shape as the
operations: a config whose `type` names an implementation in a registry. A dataset yields plain `dict`
rows. intex ships no implementation: you register your own and `DatasetFactory` resolves it by
`type`.

```python
import json
from pathlib import Path

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory


@DatasetFactory.register("jsonl")
class JsonlDataset(AbstractDataset):
    def load(self) -> list[Row]:
        lines = Path(self.config.path).read_text().splitlines()  # 'path' is a property of the config
        return [json.loads(line) for line in lines][: self.config.get("limit")]


Path("prompts.jsonl").write_text('{"prompt": "The capital of France is"}\n{"prompt": "The Eiffel Tower is in"}\n')

dataset = DatasetFactory.create(DatasetConfig(type="jsonl", path="prompts.jsonl", limit=10)).prepare()
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
from intex.operation.config import ConfigFactory

config = ConfigFactory[DatasetConfig]("configs.yaml").create("my_data")
```

`DatasetConfig` declares only `type`; every other key of the entry is kept as a property, readable as
`config.path` or `config.get("limit", 100)`, so a dataset defines its own settings without intex
knowing about them. Subclass `DatasetConfig` with real dataclass fields to have settings typed and
validated instead; keys it does not declare still land in `config.properties`.

`load` is called once, lazily, by `prepare()`; `len`, iteration and indexing then read the cached
rows. Operations do not consume datasets yet: `LensInput`, `ProbeInput`, `PatchingInput` and
`CircuitInput` take their prompts directly.

Registering a dataset follows the same rules as registering a method; see
[Registration](custom-methods.md#registration).
