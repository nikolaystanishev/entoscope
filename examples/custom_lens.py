"""Register a lens of your own and run it from custom_lens.yaml.

    uv run python examples/custom_lens.py --config-id warm

The lens is registered when this module runs, above ``main``: an experiment script runs ``main`` as
it is decorated, and a config naming the lens is checked against the registry as it is read.
"""

from dataclasses import dataclass, field
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from intex.lens.config import LensConfig, LensMethodConfig
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.logit_lens import LogitLens
from intex.operation.config import ConfigFactory
from intex.script import experiment

CONFIGS = Path(__file__).with_name("custom_lens.yaml")


@dataclass(kw_only=True)
class TemperatureLensConfig(LensMethodConfig):
    temperature: float = 1.0


@LensMethodFactory.register("temperature_lens")
class TemperatureLens(LogitLens[TemperatureLensConfig]):
    """The logit lens, with its logits divided by a temperature before the collector's softmax."""

    def project(self, activation, index):
        return super().project(activation, index) / self.method_config.temperature


@dataclass
class Config:
    config_id: str = field(default="warm", metadata={"help": "the entry of the config file to run"})
    config_path: str = field(default=str(CONFIGS), metadata={"help": "a YAML file"})
    prompt: str = "The capital of France is"
    model_name: str = "gpt2"
    output_dir: str = field(default="runs", metadata={"help": "results go into <output-dir>/<config-id>"})
    dry_run: bool = field(default=False, metadata={"help": "log what the run would do, without running it"})


@experiment
def main(config: Config):
    """Run the temperature lens on a prompt."""
    model = TransformerBridge.boot_transformers(config.model_name)
    model.eval()

    lens_input = LensInput(ConfigFactory[LensConfig](config.config_path).create(config.config_id), config.prompt)
    save_dir = f"{config.output_dir}/{config.config_id}"

    if config.dry_run:
        Lens(model).plan(lens_input, save_dir)
    else:
        Lens(model).apply(lens_input, save_dir)
