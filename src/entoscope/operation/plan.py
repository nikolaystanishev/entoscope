from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from entoscope.operation.config import AbstractConfig

MAX_LISTED = 10


def _listed(items: list[str]) -> str:
    if len(items) <= MAX_LISTED:
        return ", ".join(items)
    return f"{', '.join(items[:MAX_LISTED])}, +{len(items) - MAX_LISTED} more"


def _printable(index: pd.DataFrame) -> pd.DataFrame:
    """Spell an index's locations as ``index.csv`` does, since ``to_string`` prints an enum by its member name.

    Args:
        index: An index frame.

    Returns:
        The frame with every enum cell replaced by its value.
    """
    return index.map(lambda cell: cell.value if isinstance(cell, Enum) else cell)


def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {noun}{'es' if noun.endswith(('s', 'x', 'ch', 'sh')) else 's'}"


@dataclass
class OperationPlan:
    """What a run would do, as an operator's ``plan`` reports it.

    ``str(plan)`` gives ``describe()``.

    Attributes:
        operation: The operation's name, such as ``"Lens"``.
        method: The name of the method the config runs.
        config: The config the run would be made under.
        steps: How many units of work the run makes.
        steps_label: What a unit of work is: an index element, a probe, a patched run or a prompt.
        forward_passes: Every forward pass the run makes, the cached runs of the prompts included.
        backward_passes: Every backward pass the run makes.
        batches: How many batches the run makes, set only for a method that runs the model a batch at a
            time.
        index: One frame per index, the same frame ``index.csv`` would hold.
        prompts: One ``(label, prompt, token count)`` per prompt.
        activations: The transformer activation names the run caches.
        reads_full_cache: Whether the run caches every activation, in which case ``activations`` is empty.
        save_dir: The directory the run would write into.
        outputs: The file names the run would write into ``save_dir``.
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
        """Render the plan as text, which is what ``str(plan)`` and the log show.

        Args:
            max_rows: The most rows of each index to print.

        Returns:
            The plan, one fact per line, followed by each index.
        """
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
