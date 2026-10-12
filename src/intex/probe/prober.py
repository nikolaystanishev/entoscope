import logging

from tqdm import tqdm

from intex.model import Model
from intex.operation.operator import Operator
from intex.operation.plan import OperationPlan
from intex.probe.collector import ProbeCollector
from intex.probe.input import ProbeInput
from intex.probe.method.factory import ProbeMethodFactory
from intex.probe.metric import AbstractProbeMetric

logger = logging.getLogger(__name__)


class Prober(Operator):
    """Trains one probe per location and scores it on the training examples (no held-out split).

    Each example prompt is run once, caching only what the index reads. The probing collector draws no
    page: its result is a table of scores and the fitted probes.
    """

    def __init__(self, model: Model, collector_class: type[ProbeCollector] = ProbeCollector):
        self.model = model
        self.collector_class = collector_class

    def apply(
        self,
        probe_input: ProbeInput,
        save_dir: str | None = None,
        return_collector: bool = False,
        verbose: bool = True,
    ) -> ProbeCollector | None:
        # Built as every location builds its own, so a probe that cannot be built fails before the forward passes.
        ProbeMethodFactory.create(probe_input.config, self.model)
        probe_input.prepare(self.model)
        collector = self.collector_class(probe_input, self.model, save_dir)
        collector.prepare()
        labels = probe_input.labels.detach().cpu().numpy()

        for single_index in tqdm(probe_input.index, total=len(probe_input.index), disable=not verbose):
            features = probe_input.features[single_index.output]
            probe = ProbeMethodFactory.create(probe_input.config, self.model)
            probe.fit(features, probe_input.labels)
            predictions = probe.predict(features)
            probabilities = probe.predict_proba(features)
            metric_values = {
                metric: AbstractProbeMetric.get(metric).compute(labels, predictions, probabilities)
                for metric in probe_input.config.metric
            }
            collector.collect(probe, metric_values, single_index.output)

        collector.save()

        if return_collector:
            return collector
        return None

    def plan(self, probe_input: ProbeInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        # Built as the run builds them, so a plan fails wherever its run would.
        ProbeMethodFactory.create(probe_input.config, self.model)
        probe_input.prepare_index(self.model)

        prompts = [("positive", prompt) for prompt in probe_input.positive_prompts]
        prompts += [("negative", prompt) for prompt in probe_input.negative_prompts]

        plan = OperationPlan(
            operation="Prober",
            method=probe_input.config.method.type,
            config=probe_input.config,
            steps=len(probe_input.index),
            steps_label="probe",
            # One cached pass per example: features are collected a prompt at a time.
            forward_passes=len(prompts),
            index=[probe_input.index.to_df()],
            prompts=[
                (label, prompt, tokens.shape[1])
                for (label, prompt), tokens in zip(prompts, probe_input.token_tensors, strict=True)
            ],
            activations=sorted(probe_input.index.source_activation_names()),
            save_dir=save_dir,
            outputs=self.collector_class(probe_input, self.model, save_dir).output_files(),
        )

        if verbose:
            logger.info("%s", plan)
        return plan
