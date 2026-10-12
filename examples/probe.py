"""Run one entry of probe.yaml on positive and negative example prompts.

    uv run python examples/probe.py --config-id last_token_by_layer
    uv run python examples/probe.py --positive "I loved it" "Great fun" --negative "I hated it" "Dull"

Results land in runs/<config-id>/: the scores in result.csv, the fitted probes in probes.pkl.
"""

from dataclasses import dataclass, field
from pathlib import Path

from transformer_lens.model_bridge import TransformerBridge

from intex.model import CachedModel
from intex.operation.config import ConfigFactory
from intex.probe.config import ProbeConfig
from intex.probe.input import ProbeInput
from intex.probe.prober import Prober
from intex.script import experiment

CONFIGS = Path(__file__).with_name("probe.yaml")

POSITIVE = [
    "The movie was absolutely wonderful",
    "I loved every minute of it",
    "What a delightful surprise this was",
    "The food was fresh and delicious",
    "She gave a brilliant performance",
    "This is the best day of my life",
    "The staff were kind and helpful",
    "I would happily do it again",
]
NEGATIVE = [
    "The movie was dreadful and boring",
    "I hated every minute of it",
    "What a miserable waste of time",
    "The food was cold and tasteless",
    "She gave a painful performance",
    "This is the worst day of my life",
    "The staff were rude and careless",
    "I would never do it again",
]


@dataclass
class Config:
    config_id: str = field(default="residual_by_layer", metadata={"help": "the entry of the config file to run"})
    config_path: str = field(default=str(CONFIGS), metadata={"help": "a YAML file"})
    positive: list[str] = field(default_factory=lambda: list(POSITIVE), metadata={"help": "prompts labelled 1"})
    negative: list[str] = field(default_factory=lambda: list(NEGATIVE), metadata={"help": "prompts labelled 0"})
    model_name: str = "gpt2"
    output_dir: str = field(default="runs", metadata={"help": "results go into <output-dir>/<config-id>"})
    cache_dir: str | None = field(default=None, metadata={"help": "keep the model's activations on disk here"})
    dry_run: bool = field(default=False, metadata={"help": "log what the run would do, without running it"})


@experiment
def main(config: Config):
    """Train and score one probe per location of a probe config."""
    model = CachedModel(TransformerBridge.boot_transformers(config.model_name), config.cache_dir)
    model.eval()

    probe_input = ProbeInput(
        ConfigFactory[ProbeConfig](config.config_path).create(config.config_id), config.positive, config.negative
    )
    save_dir = f"{config.output_dir}/{config.config_id}"

    if config.dry_run:
        Prober(model).plan(probe_input, save_dir)
    else:
        Prober(model).apply(probe_input, save_dir)
