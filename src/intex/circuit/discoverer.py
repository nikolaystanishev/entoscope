from tqdm import tqdm

from intex.circuit.collector import CircuitCollector, CircuitVisualizationCollector
from intex.circuit.input import CircuitInput
from intex.circuit.method.factory import CircuitMethodFactory
from intex.model import Model
from intex.operation.operator import Operator


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
