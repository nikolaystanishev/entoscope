from tqdm import tqdm

from intex.lens.collector import LensCollector
from intex.lens.input import LensInput
from intex.lens.method.factory import LensMethodFactory
from intex.model import Model
from intex.operation.operator import Operator


class Lens(Operator):
    def __init__(self, model: Model, collector_class: type[LensCollector] = LensCollector):
        self.model = model
        self.collector_class = collector_class

    def apply(
        self,
        lens_input: LensInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> LensCollector | None:
        lens_input.prepare(self.model)
        method = LensMethodFactory.create(lens_input.config, self.model)
        collector = self.collector_class(lens_input, self.model, save_dir)
        collector.prepare()

        for single_index in tqdm(lens_input.index, total=len(lens_input.index), disable=not verbose):
            activation = lens_input.cache[single_index.source.transformer_activation()][single_index.source.slices()]
            collector.collect(method.project(activation, single_index.source), single_index.output)

        collector.save()

        if return_collector:
            return collector
        return None
