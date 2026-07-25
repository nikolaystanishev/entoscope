from tqdm import tqdm

from intex.circuit.collector import CircuitCollector, CircuitVisualizationCollector
from intex.circuit.input import CircuitInput
from intex.circuit.method.factory import CircuitMethodFactory
from intex.model import Model
from intex.operation.operator import Operator
from intex.operation.plan import OperationPlan


class CircuitDiscoverer(Operator):
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
        circuit_input.prepare(self.model)
        method = CircuitMethodFactory.create(circuit_input.config, self.model)
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
        circuit_input.prepare_index(self.model)

        plan = OperationPlan(
            operation="CircuitDiscoverer",
            config=circuit_input.config,
            steps=len(circuit_input.prompts),
            steps_label="prompt",
            # The method rebuilds the whole cache for each prompt, so there is no index to narrow
            # and no names to filter by — the graph is built over every activation.
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
            print(plan)
        return plan
