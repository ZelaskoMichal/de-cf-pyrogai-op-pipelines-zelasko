"""Log metadata step."""
import os

import joblib
import pandas as pd
import vertexai
from google.cloud import aiplatform

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step


class LogMetadata(Step):
    """Step class logging the metadata to Vertex AI."""

    def __init__(self, *args, **kwargs):
        """Custom step initialization."""
        super().__init__(*args, **kwargs)
        vertexai.init(
            project=self.config["vertex_meta"]["project_name"],
            experiment=self.config["vertex_meta"]["experiment"],
            staging_bucket=self.config["vertex_meta"]["staging_bucket"],
        )

    def _load_data_and_model(self):
        path_to_df = self.ioctx.get_fn("dataframe/test_df.csv")
        df = pd.read_csv(path_to_df)
        path_to_model = self.ioctx.get_fn("model/linear_regression_model.joblib")
        model = joblib.load(path_to_model)
        return df, model

    def _log_run(self, my_run, model, df):
        metrics = self.inputs["metrics"]
        params = self.inputs["params"]
        aiplatform.log_metrics(metrics)
        aiplatform.log_params(params)
        my_run.log_model(
            model,
            input_example=df,
            display_name="example linear regression model",
        )

    def run(self):
        """Running code logic."""
        df, model = self._load_data_and_model()
        if self.platform == Platform.VERTEX:
            with aiplatform.start_run(os.environ["PIPELINE_JOB_NAME"]) as my_run:
                self.logger.info(f"Aiplatform run id: {os.environ['PIPELINE_JOB_NAME']}")
                self._log_run(my_run, model, df)

        else:
            with aiplatform.start_run("vertex-meta-test-run", resume=True) as my_run:
                self._log_run(my_run, model, df)
