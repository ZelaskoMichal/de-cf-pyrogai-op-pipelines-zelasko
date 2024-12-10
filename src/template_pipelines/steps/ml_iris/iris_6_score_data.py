"""ScoreDataStep class."""
import json
import pickle

import pandas as pd

from aif.pyrogai.steps.step import Step


class ScoreDataStep(Step):
    """Scores data.

    Scoring includes:
    - predict clusters for test data
    - calculate model score for test data
    """

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn_test_data = self.ioctx.get_fn("test_data.parquet")
        data = pd.read_parquet(fn_test_data).reset_index()
        self.logger.info(f"Input row count {len(data)}")

        fn_trained_model = self.ioctx.get_fn("trained_model.pickle")
        with open(fn_trained_model, "rb") as f:
            model = pickle.load(f)

        fn_model_results = self.ioctx.get_fn("model_results.pickle")
        with open(fn_model_results, "r") as f:
            config = json.load(f)

        # define local variables names for convenience
        target = self.config["ml_iris"]["target"]
        predicted = f"{target}_predicted"
        features = self.config["ml_iris"]["features"]

        # add predictions
        data[predicted] = model.predict(data[features])

        # calculate score
        config["test_score"] = len(data[data[target] == data[predicted]]) / len(data)

        self.logger.info(f"Test Score {config['test_score']}")

        # save output with iocontext
        fn_test_predicted = self.ioctx.get_output_fn("test_predicted.parquet")
        data.to_parquet(fn_test_predicted)

        fn_model_results = self.ioctx.get_output_fn("model_results.json")
        with open(fn_model_results, "w") as f:
            json.dump(config, f)

        self.logger.info("Iris data scored.")
