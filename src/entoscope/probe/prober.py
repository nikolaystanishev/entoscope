import logging

from tqdm import tqdm

from entoscope.model import Model
from entoscope.operation.operator import Operator
from entoscope.operation.plan import OperationPlan
from entoscope.probe.collector import ProbeCollector
from entoscope.probe.input import ProbeInput
from entoscope.probe.method.factory import ProbeMethodFactory
from entoscope.probe.metric import AbstractProbeMetric

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
        ProbeMethodFactory.create(self.model, probe_input.config)
        probe_input.prepare(self.model)
        collector = self.collector_class(probe_input, self.model, save_dir)
        labels = probe_input.labels.detach().cpu().numpy()

        for single_index in tqdm(probe_input.index, total=len(probe_input.index), disable=not verbose):
            features = probe_input.features[single_index.output]
            probe = ProbeMethodFactory.create(self.model, probe_input.config)
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
        ProbeMethodFactory.create(self.model, probe_input.config)
        probe_input.prepare_index(self.model)

        prompts = [("positive", prompt) for prompt in probe_input.positive_prompts]
        prompts += [("negative", prompt) for prompt in probe_input.negative_prompts]

        plan = OperationPlan(
            operation="Prober",
            method=probe_input.config.method.type,
            config=probe_input.config,
            steps=len(probe_input.index),
            steps_label="probe",
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
