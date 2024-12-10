"""Example of sweep step trials and subsequent step which consumes best trial.

Example from based on
https://github.com/Azure/azureml-examples/blob/dd15e3f7d6a512fedfdfbdb4be19e065e8c1d224/
 sdk/python/jobs/single-step/lightgbm/iris/lightgbm-iris-sweep.ipynb
"""
import json
from tempfile import NamedTemporaryFile
from typing import Tuple

import lightgbm as lgbm
import pandas as pd

from aif.pyrogai.ioutils.formats import read_any
from aif.pyrogai.steps import AmlSweepStep
from template_pipelines.utils.aml_sweep.io_utils import get_ioslot_name


def train_model(params, x_train, x_test, y_train, y_test):
    """Run training and test model, mlflow will log metrics using self.mlflow.autolog() ."""
    train_data = lgbm.Dataset(x_train, label=y_train)
    test_data = lgbm.Dataset(x_test, label=y_test)

    model = lgbm.train(
        params,
        train_data,
        valid_sets=[test_data],
        valid_names=["test"],
    )
    return model


class MyAmlSweepStep(AmlSweepStep):
    """Sweep step, run method defines trial content.

    Trial parameters are taken from pipeline params, updated with search space params from the config.
    Best trial will be evaluated on a metric logged to mlflow.
    By default, AmlSweepStep runs with self.mlflow.autolog() to log automatically,
    this can be augmented by logging additional metrics.
    """

    def run(self) -> None:
        """Trial logic goes here."""
        # remember that pipeline parameters are passed as string, need to cast to proper types as below!
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "boosting_type": self.runtime_parameters["boosting_type"],
            "num_iterations": int(self.runtime_parameters["num_iterations"]),
            "max_leaf_nodes": int(self.runtime_parameters["max_leaf_nodes"]),
            "learning_rate": float(self.runtime_parameters["learning_rate"]),
            "metric": self.runtime_parameters["metric"],
            "random_seed": int(self.runtime_parameters["random_seed"]),
            "verbose": int(self.runtime_parameters["verbose"]),
        }

        x_train, x_test, y_train, y_test = self._get_data()

        # train model
        model = train_model(params, x_train, x_test, y_train, y_test)

        # save outputs
        # whatever we want to pass on to the consumer from the best run must be saved in AML IOslot(s)
        # when run on AML, only the best trial's output will be exposed to consumers
        # when run locally, there will only be 1 run (at the values of the runtime parameters)
        # since local runs do not have AML IOslots, we use cloudfile IOslot for local development
        with NamedTemporaryFile() as tf:
            # Produce model to a temporary file
            model.save_model(tf.name)
            # save it to the io slot
            self.outputs[get_ioslot_name(self, "trial_model")] = tf.name
        with NamedTemporaryFile() as tf:
            # Produce parameters to a temporary file
            with open(tf.name, "w") as fp:
                json.dump(params, fp)
            # save it to the io slot
            self.outputs[get_ioslot_name(self, "trial_params")] = tf.name

    def _get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.debug(f"reading train from IOcontext 'data/train.parquet'")
        train = read_any(self.ioctx.get_fn("data/train.parquet"))
        x_train, y_train = self._split_x_y(train)
        self.logger.debug(f"reading test from IOcontext 'data/test.parquet'")
        test = read_any(self.ioctx.get_fn("data/test.parquet"))
        x_test, y_test = self._split_x_y(test)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def _split_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y_name = "species"
        x_df = df.drop([y_name], axis=1)
        y_df = df[y_name]
        return x_df, y_df
