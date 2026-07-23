import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any

from intex.operation.config import AbstractConfig
from intex.util.data import from_dict


@dataclass(kw_only=True, init=False)
class DatasetConfig(AbstractConfig):
    """``type`` selects the registered dataset; every other key is a dynamic property of it."""

    type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *, type: str, config_id: str | None = None, properties: dict[str, Any] | None = None, **extra):
        self.config_id = config_id
        self.type = type
        self.properties = {**(properties or {}), **extra}

    @classmethod
    def from_config(cls, value: Any) -> "DatasetConfig":
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
        if "type" not in value:
            raise ValueError(f"Missing required keys for {cls.__name__}: ['type']")

        declared = {
            declared_field.name
            for declared_field in dataclasses.fields(cls)
            if declared_field.init and declared_field.name != "properties"
        }
        hints = typing.get_type_hints(cls)
        known = {name: from_dict(hints[name], element) for name, element in value.items() if name in declared}
        properties = {name: element for name, element in value.items() if name not in declared}

        return cls(**known, properties=properties)

    def __getattr__(self, name: str) -> Any:
        properties = self.__dict__.get("properties")
        if properties is None or name not in properties:
            raise AttributeError(f"Dataset config '{self.__dict__.get('type')}' has no property '{name}'")
        return properties[name]

    def __contains__(self, name: str) -> bool:
        return name in self.properties

    def get(self, name: str, default: Any = None) -> Any:
        return self.properties.get(name, default)
