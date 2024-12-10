"""Consumer step for AML sweep."""

import json

import lightgbm as lgb

from aif.pyrogai.ioutils import read_any
from aif.pyrogai.steps.step import Step
from template_pipelines.utils.aml_sweep.io_utils import get_ioslot_name


class SweepConsumer(Step):
    """Example step to fetch output of sweep step execution."""

    def run(self) -> None:
        """Sample next step that fetches best trial's output."""
        with open(self.inputs[get_ioslot_name(self, "trial_params")]) as fh:
            # print the best hyperparameters
            self.logger.info("best trial's params IOslot content:")
            params = json.load(fh)
            self.logger.info(params)

        # get the model and use it to make predictions
        model = lgb.Booster(model_file=self.inputs[get_ioslot_name(self, "trial_model")])
        self.logger.debug(f"reading test from IOcontext 'data/test.parquet'")
        test = read_any(self.ioctx.get_fn("data/test.parquet"))
        pred = model.predict(test.drop(columns="species"))
        self.logger.info("Test predicitons:")
        self.logger.info(pred)
