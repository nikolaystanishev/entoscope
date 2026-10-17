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
    per prompt from a forward pass of that prompt. The default collector also draws a
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
        method = CircuitMethodFactory.create(self.model, circuit_input.config)
        circuit_input.prepare(self.model)
        collector = self.collector_class(circuit_input, self.model, save_dir)

        for i, prompt in enumerate(tqdm(circuit_input.prompts, disable=not verbose)):
            collector.collect(method.build(prompt, self.model.to_tokens(prompt)), i)

        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, circuit_input: CircuitInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        method = CircuitMethodFactory.create(self.model, circuit_input.config)
        circuit_input.prepare_index(self.model)
        activations = method.activation_names()

        plan = OperationPlan(
            operation="CircuitDiscoverer",
            method=circuit_input.config.method.type,
            config=circuit_input.config,
            steps=len(circuit_input.prompts),
            steps_label="prompt",
            forward_passes=len(circuit_input.prompts),
            prompts=[
                (f"prompt {i}", prompt, self.model.to_tokens(prompt).shape[1])
                for i, prompt in enumerate(circuit_input.prompts)
            ],
            activations=activations or [],
            reads_full_cache=activations is None,
            save_dir=save_dir,
            outputs=self.collector_class(circuit_input, self.model, save_dir).output_files(),
        )

        if verbose:
            logger.info("%s", plan)
        return plan
