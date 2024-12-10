"""TrainModelStep class."""
import json
import pickle

import pandas as pd
from sklearn.svm import SVC

from aif.pyrogai.steps.step import Step


class TrainModelStep(Step):
    """Trains model."""

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("train_data.parquet")
        data = pd.read_parquet(fn).reset_index()
        self.logger.info(f"Input row count {len(data)}")

        # define local variables names for convenience
        target = self.config["ml_observability"]["target"]
        features = self.config["ml_observability"]["features"]
        predicted = f"{self.config['ml_observability']['target']}_predicted"
        config = {}

        # train model
        model = SVC()
        model.fit(data[features], data[target])

        # add predictions
        data[predicted] = model.predict(data[features])

        # calculate score
        config["train_score"] = len(data[data[target] == data[predicted]]) / len(data)

        # save output with iocontext
        fn_data = self.ioctx.get_output_fn("train_predicted.parquet")
        data.to_parquet(fn_data)

        fn_trained_model = self.ioctx.get_output_fn("trained_model.pickle")
        with open(fn_trained_model, "wb") as f:
            pickle.dump(model, f)

        fn_model_results = self.ioctx.get_output_fn("model_results.pickle")
        with open(fn_model_results, "w") as f:
            json.dump(config, f)

        self.logger.info("Iris model trained.")
