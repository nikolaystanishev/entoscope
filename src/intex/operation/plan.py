from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from intex.operation.config import AbstractConfig

# Prompts and output files are enumerated per prompt, so a run over many of them would bury the
# rest of the summary; the index keeps its own (much larger) budget through ``describe``.
MAX_LISTED = 10


def _listed(items: list[str]) -> str:
    if len(items) <= MAX_LISTED:
        return ", ".join(items)
    return f"{', '.join(items[:MAX_LISTED])}, +{len(items) - MAX_LISTED} more"


def _printable(index: pd.DataFrame) -> pd.DataFrame:
    """Locations as ``index.csv`` spells them - ``to_string`` would print an enum by its member name."""
    return index.map(lambda cell: cell.value if isinstance(cell, Enum) else cell)


def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {noun}{'es' if noun.endswith(('s', 'x', 'ch', 'sh')) else 's'}"


@dataclass
class OperationPlan:
    """What a run would do, as an operator's ``plan`` reports it.

    ``steps`` counts the run's units of work, named by ``steps_label`` (an index element, a probe, a
    patched run, a prompt). ``forward_passes`` and ``backward_passes`` count every pass, the cached runs
    of the prompts included; ``batches`` is set only for a method that runs the model a batch at a time.
    ``index`` holds one frame per index, the same frame ``index.csv`` would hold, and ``prompts`` one
    ``(label, prompt, token count)`` per prompt. ``activations`` are the hook names the run caches, unless
    ``reads_full_cache`` says it caches every activation. ``outputs`` are the file names the run would
    write into ``save_dir``.
    """

    operation: str
    method: str
    config: AbstractConfig
    steps: int
    steps_label: str
    forward_passes: int
    backward_passes: int = 0
    batches: int | None = None
    index: list[pd.DataFrame] = field(default_factory=list)
    prompts: list[tuple[str, str, int]] = field(default_factory=list)
    activations: list[str] = field(default_factory=list)
    reads_full_cache: bool = False
    save_dir: str | None = None
    outputs: list[str] = field(default_factory=list)

    def describe(self, max_rows: int = 100) -> str:
        """The plan as text, with up to ``max_rows`` rows of each index - what ``str(plan)`` and the log show."""
        lines = [f"{self.operation} plan", f"  method: {self.method}"]
        if self.config.config_id:
            lines.append(f"  config: {self.config.config_id}")
        for label, prompt, n_tokens in self.prompts[:MAX_LISTED]:
            lines.append(f'  {label}: "{prompt}" ({_plural(n_tokens, "token")})')
        if len(self.prompts) > MAX_LISTED:
            lines.append(f"  +{len(self.prompts) - MAX_LISTED} more prompts")
        if self.reads_full_cache:
            lines.append("  reads: full activation cache")
        elif self.activations:
            lines.append(f"  reads: {_listed(self.activations)}")
        lines.append(f"  work: {self._work()}")
        lines.append(f"  writes: {self._writes()}")
        for i, index in enumerate(self.index):
            name = "index" if len(self.index) == 1 else f"index {i}"
            lines.append(f"  {name} ({_plural(len(index), 'row')}):")
            lines.append(textwrap.indent(_printable(index).to_string(index=False, max_rows=max_rows), "    "))

        return "\n".join(lines)

    def _work(self) -> str:
        work = _plural(self.steps, self.steps_label)
        if self.batches is not None:
            work = f"{work} in {_plural(self.batches, 'batch')}"
        passes = _plural(self.forward_passes, "forward pass")
        if self.backward_passes:
            passes = f"{passes}, {_plural(self.backward_passes, 'backward pass')}"
        return f"{work}, {passes}"

    def _writes(self) -> str:
        if self.save_dir is None:
            return "nothing (no save_dir)"
        return f"{self.save_dir}/{{{_listed(self.outputs)}}}"

    def __str__(self) -> str:
        return self.describe()
