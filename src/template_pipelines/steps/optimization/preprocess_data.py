"""Preprocess data and create resulting datasets (Standard Data Model - SDM)."""

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM


class PreprocessData(Step):
    """PreprocessData step."""

    def pre_run(self) -> None:
        """Keep mlflow run ids to testing purpose."""
        self.active_mflow_run_id = self.mlflow_utils.get_active_run_id()
        self.root_mlflow_run_id = self.mlflow_utils.get_root_run_id()

    def run(self):
        """Entry point for preprocessing step."""
        sdm = StocksPortfolioSDM()
        metrics = sdm.create(data_path=f"{self.config['input_tmp_dir']}", ioctx=self.ioctx)
        sdm.save(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)

        # log metrics directly to current step
        self.mlflow_utils.log(metrics=metrics)
