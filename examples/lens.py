"""Run one entry of lens.yaml on a prompt.

    uv run python examples/lens.py --config-id tuned_lens
    uv run python examples/lens.py --config-id parameter_lens --dry-run

Results land in runs/<config-id>/: open result.html in a browser.
"""

from dataclasses import dataclass, field
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from entoscope import CachedModel, ConfigFactory, Lens, LensConfig, LensInput, experiment
from entoscope.lens import LensMethodFactory

CONFIGS = Path(__file__).with_name("lens.yaml")


@dataclass
class Config:
    config_id: str = field(default="logit_lens", metadata={"help": "the entry of the config file to run"})
    config_path: str = field(default=str(CONFIGS), metadata={"help": "a YAML file"})
    prompt: str = "The capital of France is"
    model_name: str = "gpt2"
    output_dir: str = field(default="runs", metadata={"help": "results go into <output-dir>/<config-id>"})
    cache_dir: str | None = field(default=None, metadata={"help": "keep the model's activations on disk here"})
    dry_run: bool = field(default=False, metadata={"help": "log what the run would do, without running it"})


@experiment
def main(config: Config):
    """Run one lens config on a prompt."""
    model = CachedModel(TransformerBridge.boot_transformers(config.model_name), config.cache_dir)
    model.eval()

    lens_config = ConfigFactory[LensConfig](config.config_path).create(config.config_id)
    # A lens that reads the weights rather than the activations takes no prompt.
    reads_activations = LensMethodFactory.get(lens_config.method.type).reads_activations
    lens_input = LensInput(lens_config, config.prompt if reads_activations else None)
    save_dir = f"{config.output_dir}/{config.config_id}"

    if config.dry_run:
        Lens(model).plan(lens_input, save_dir)
    else:
        Lens(model).apply(lens_input, save_dir)
