import logging

from tqdm import tqdm

from entoscope.circuit.collector import CircuitCollector, CircuitVisualizationCollector
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.factory import CircuitMethodFactory
from entoscope.model import Model
from entoscope.operation.operator import Operator
from entoscope.operation.plan import OperationPlan

logger = logging.getLogger(__name__)


class CircuitDiscoverer(Operator):
    """Discovers the circuit behind each prompt with the method ``config.method`` names.

    There is no index: a circuit spans every layer, position and head, so the method builds one graph
    per prompt from a full forward pass of that prompt. The default collector also draws a
    ``circuit_{i}.html`` per prompt; ``CircuitCollector`` writes the graph files alone.
    """

    def __init__(
        self,
        model: Model,
        collector_class: type[CircuitCollector] = CircuitVisualizationCollector,
    ):
        self.model = model
        self.collector_class = collector_class

    def apply(
        self,
        circuit_input: CircuitInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> CircuitCollector | None:
        # Built before the prompts are prepared, so a method that cannot be built fails before the run does any work.
        method = CircuitMethodFactory.create(circuit_input.config, self.model)
        circuit_input.prepare(self.model)
        collector = self.collector_class(circuit_input, self.model, save_dir)
        collector.prepare()

        prompts = list(zip(circuit_input.prompts, circuit_input.tokens, strict=True))
        for i, (prompt, tokens) in enumerate(tqdm(prompts, total=len(prompts), disable=not verbose)):
            collector.collect(method.build(prompt, tokens), i)

        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, circuit_input: CircuitInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        # Built as the run builds it, so a plan fails wherever its run would.
        CircuitMethodFactory.create(circuit_input.config, self.model)
        circuit_input.prepare_index(self.model)

        plan = OperationPlan(
            operation="CircuitDiscoverer",
            method=circuit_input.config.method.type,
            config=circuit_input.config,
            steps=len(circuit_input.prompts),
            steps_label="prompt",
            # The method rebuilds the whole cache for each prompt, so there is no index to narrow
            # and no names to filter by - the graph is built over every activation.
            forward_passes=len(circuit_input.prompts),
            prompts=[
                (f"prompt {i}", prompt, tokens.shape[1])
                for i, (prompt, tokens) in enumerate(zip(circuit_input.prompts, circuit_input.tokens, strict=True))
            ],
            reads_full_cache=True,
            save_dir=save_dir,
            outputs=self.collector_class(circuit_input, self.model, save_dir).output_files(),
        )

        if verbose:
            logger.info("%s", plan)
        return plan
