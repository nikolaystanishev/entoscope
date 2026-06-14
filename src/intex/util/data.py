import json
import os
import re

import yaml


def write_json(file_path: str, data: dict | list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_yaml(file_path: str) -> list | dict:
    loader = yaml.FullLoader
    tag = "!ENV"
    pattern = re.compile(r".*?\${(\w+)}.*?")
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(yaml_loader: yaml.FullLoader, node):
        value = yaml_loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                key = f"${{{g}}}"
                env_value = os.environ.get(g, g)
                if key == full_value and env_value.isdigit():
                    full_value = int(env_value)
                else:
                    full_value = full_value.replace(key, env_value)
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(file_path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def create_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
