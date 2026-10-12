"""Run one entry of circuit.yaml on some prompts.

    uv run python examples/circuit.py
    uv run python examples/circuit.py --config-id strong_routes --prompts "The Eiffel Tower is in"

Results land in runs/<config-id>/: open circuit_<i>.html in a browser, one page per prompt.
"""

from dataclasses import dataclass, field
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.model import CachedModel
from entoscope.operation.config import ConfigFactory
from entoscope.script import experiment

CONFIGS = Path(__file__).with_name("circuit.yaml")


@dataclass
class Config:
    config_id: str = field(default="information_flow_routes", metadata={"help": "the entry of the config file to run"})
    config_path: str = field(default=str(CONFIGS), metadata={"help": "a YAML file"})
    prompts: list[str] = field(
        default_factory=lambda: ["The capital of France is", "The Eiffel Tower is in"],
        metadata={"help": "one circuit is discovered per prompt"},
    )
    model_name: str = "gpt2"
    output_dir: str = field(default="runs", metadata={"help": "results go into <output-dir>/<config-id>"})
    cache_dir: str | None = field(default=None, metadata={"help": "keep the model's activations on disk here"})
    dry_run: bool = field(default=False, metadata={"help": "log what the run would do, without running it"})


@experiment
def main(config: Config):
    """Discover the circuit behind each prompt."""
    model = CachedModel(TransformerBridge.boot_transformers(config.model_name), config.cache_dir)
    model.eval()

    circuit_input = CircuitInput(
        ConfigFactory[CircuitConfig](config.config_path).create(config.config_id), config.prompts
    )
    save_dir = f"{config.output_dir}/{config.config_id}"

    if config.dry_run:
        CircuitDiscoverer(model).plan(circuit_input, save_dir)
    else:
        CircuitDiscoverer(model).apply(circuit_input, save_dir)
