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
    per prompt from a forward pass of that prompt.

    Args:
        model: The model to read.
        collector_class: The collector to write the results with. The default also draws a
            ``circuit_{i}.html`` per prompt, and ``CircuitCollector`` writes the graph files alone.
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
        """Build the circuit behind every prompt of the input.

        The method is built before any forward pass, so a config that names something the run cannot do
        fails before the model is touched. Where the results were saved is logged through the
        ``entoscope`` logger.

        Args:
            circuit_input: The config and prompts to run.
            save_dir: The directory to write every prompt's graph files and page, ``prompts.json`` and
                ``config.json`` into, created if missing. ``None`` writes nothing.
            return_collector: Whether to return the collector.
            verbose: Whether to show a progress bar.

        Returns:
            The collector, whose ``result`` holds one ``PromptCircuit`` per prompt, if
            ``return_collector`` is set, and ``None`` otherwise.
        """
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
        """Describe the run ``apply`` would make, without a forward pass and without touching ``save_dir``.

        Args:
            circuit_input: The config and prompts the run would use.
            save_dir: The directory the run would write into.
            verbose: Whether to also log the plan.

        Returns:
            The plan: one step and one forward pass per prompt.
        """
        method = CircuitMethodFactory.create(self.model, circuit_input.config)
        circuit_input.prepare(self.model, index_only=True)
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
