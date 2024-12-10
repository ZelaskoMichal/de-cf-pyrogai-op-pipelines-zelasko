"""Retrieve metadata step."""
import os

import vertexai
from google.cloud import aiplatform

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step


class RetrieveMetadata(Step):
    """Data retrieval step."""

    def __init__(self, *args, **kwargs):
        """Custom step initialization."""
        super().__init__(*args, **kwargs)
        vertexai.init(
            project=self.config["vertex_meta"]["project_name"],
            experiment=self.config["vertex_meta"]["experiment"],
            staging_bucket=self.config["vertex_meta"]["staging_bucket"],
        )

    def _get_experiment_run(self):
        if self.platform == Platform.VERTEX:
            return aiplatform.ExperimentRun(
                run_name=os.environ["PIPELINE_JOB_NAME"], experiment="template-meta"
            )
        else:
            return aiplatform.ExperimentRun(
                run_name="vertex-meta-test-run", experiment="template-meta"
            )

    @staticmethod
    def _get_experiment_data(experiment_run):
        metrics = experiment_run.get_metrics()
        parameters = experiment_run.get_params()
        model_list = experiment_run.get_experiment_models()
        experiment_model = model_list[0] if model_list is not None else None
        loaded_model = experiment_model.load_model() if experiment_model is not None else None
        return metrics, parameters, model_list, loaded_model

    def run(self):
        """Run method."""
        experiment_run = self._get_experiment_run()
        metrics, parameters, model_list, loaded_model = self._get_experiment_data(experiment_run)
        self.logger.info(f"Models logged to this run: {model_list}")
        self.logger.info(f"Metrics logged to this run: {metrics}")
        self.logger.info(f"Parameters logged to this run: {parameters}")
        self.logger.info(f"Sklearn model loaded back from the run: {loaded_model}")
