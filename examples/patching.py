"""Run one entry of patching.yaml on a pair of prompts.

    uv run python examples/patching.py --config-id head_sweep
    uv run python examples/patching.py --config-id zero_ablation --source-prompt ""
    uv run python examples/patching.py --config-id attribution --dry-run

Results land in runs/<config-id>/: open result.html in a browser. Prompts of your own come with
answers of their own, named for the metrics as --tokens correct:" Rome" distractor:" Madrid" other:" Paris".
"""

from dataclasses import dataclass, field
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from entoscope import CachedModel, ConfigFactory, Patcher, PatchingConfig, PatchingInput, experiment

CONFIGS = Path(__file__).with_name("patching.yaml")


@dataclass
class Config:
    config_id: str = field(default="residual_sweep", metadata={"help": "the entry of the config file to run"})
    config_path: str = field(default=str(CONFIGS), metadata={"help": "a YAML file"})
    source_prompt: str = field(
        default="The capital of France is", metadata={"help": "the run patched from; empty for none"}
    )
    target_prompt: str = field(default="The capital of Spain is", metadata={"help": "the run patched into"})
    tokens: dict[str, str] = field(
        default_factory=lambda: {"correct": " Paris", "distractor": " Madrid", "other": " Rome"},
        metadata={
            "help": "the tokens the metrics read, as name:token, by the names the configs give them; "
            "they replace the defaults, so name every token the config reads"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "patched runs per forward pass ('exact' only)"})
    model_name: str = "gpt2"
    output_dir: str = field(default="runs", metadata={"help": "results go into <output-dir>/<config-id>"})
    cache_dir: str | None = field(default=None, metadata={"help": "keep the model's activations on disk here"})
    dry_run: bool = field(default=False, metadata={"help": "log what the run would do, without running it"})


@experiment
def main(config: Config):
    """Run one patching config on a source and a target prompt."""
    model = CachedModel(TransformerBridge.boot_transformers(config.model_name), config.cache_dir)
    model.eval()

    patching_input = PatchingInput(
        ConfigFactory[PatchingConfig](config.config_path).create(config.config_id),
        # A source that writes a value or noise reads no source run, so it may go without a prompt.
        config.source_prompt or None,
        config.target_prompt,
        tokens=config.tokens,
    )
    patcher = Patcher(model, batch_size=config.batch_size)
    save_dir = f"{config.output_dir}/{config.config_id}"

    if config.dry_run:
        patcher.plan(patching_input, save_dir)
    else:
        patcher.apply(patching_input, save_dir)
