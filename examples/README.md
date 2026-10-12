# Examples

One experiment script per operation, each running an entry of the YAML file of configs beside it.
From the repository root:

```bash
uv run python examples/lens.py                              # the file's default entry
uv run python examples/lens.py --config-id tuned_lens       # another entry
uv run python examples/patching.py --config-id head_sweep --dry-run   # plan it, without running the model
uv run python examples/lens.py --help                       # every option
```

A run writes its results into `runs/<config-id>/`; open the `.html` page there in a browser. Every
script also takes `--model-name` (default `gpt2`), `--output-dir` (default `runs`) and `--config-path`,
and all but `custom_lens.py` take `--cache-dir` to keep the model's activations on disk between runs.

| Script | Configs | Entries |
|---|---|---|
| `lens.py` | `lens.yaml` | `logit_lens`, `neuron_logit_lens`, `head_logit_lens`, `tuned_lens`, `parameter_lens`, `subhead_parameter_lens`, `explicit_coordinates` |
| `probe.py` | `probe.yaml` | `residual_by_layer`, `last_token_by_layer`, `head_outputs` |
| `patching.py` | `patching.yaml` | `residual_sweep`, `head_sweep`, `zero_ablation`, `noise_tracing`, `attribution`, `integrated_gradients`, `explicit_coordinates`, `heads_with_fixed_mlp` |
| `circuit.py` | `circuit.yaml` | `information_flow_routes`, `strong_routes`, `last_two_predictions` |
| `custom_lens.py` | `custom_lens.yaml` | `warm`, `plain`: a lens the script registers itself |

The YAML files explain every entry in their comments. The prompts are options too: `--prompt` for
the lens, `--positive` and `--negative` for probing, `--source-prompt`, `--target-prompt`,
`--correct-token` and `--distractor-token` for patching, and `--prompts` for circuit discovery.

The test suite loads every entry and plans every script on each of them, so the examples keep up with
the code. See the [docs](../docs/index.md) for what each operation and option means.
