"""TrainModelStep class."""
import json
import pickle

import pandas as pd

from aif.pyrogai.steps.step import Step


class FakeDecisionTreeClassifier:
    """Fake implementation of a decision tree classifier."""

    def __init__(self, *args, **kwargs):
        """Fake implementation of a decision tree classifier."""
        pass

    def fit(*args):
        """Fake implementation of a fit method."""
        pass

    def predict(*args):
        """Fake implementation of a predict method."""
        return 0


class TrainModelStep(Step):
    """Trains model.

    Training includes:
    - fit 3-cluster k-means model to data
    - predict clusters for train data
    - figure out cluster labels
    - calculate model score for train data
    """

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("train_data.parquet")
        data = pd.read_parquet(fn).reset_index()
        self.logger.info(f"Input row count {len(data)}")

        # define local variables names for convenience
        target = self.config["ml_iris"]["target"]
        features = self.config["ml_iris"]["features"]
        predicted = f"{self.config['ml_iris']['target']}_predicted"
        config = {}

        # train model
        model = FakeDecisionTreeClassifier(max_depth=3, random_state=1)
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
