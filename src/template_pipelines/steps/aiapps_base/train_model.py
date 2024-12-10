"""TrainModelStep class."""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from aif.pyrogai.steps.step import Step


class TrainModelStep(Step):
    """Trains model."""

    @staticmethod
    def train_model(X, y, random_state=69, reg_strgth=0.1):
        """Train a machine learning model using the provided data."""
        model = LogisticRegression(random_state=random_state, C=reg_strgth)
        model.fit(X, y)
        return model

    def run(self):
        """Runs step."""
        # load input with iocontext
        data = self.ioctx.get_fn("preprocessed_data.csv")
        data_df = pd.read_csv(data)

        x_train = data_df.drop(columns=["target"])
        y_train = data_df["target"]

        # read regularization_strength and random_state params
        reg_strgth = float(self.runtime_parameters.get("regularization_strength"))
        random_state = int(self.runtime_parameters.get("random_state"))

        # model train
        model = self.train_model(x_train, y_train, random_state, reg_strgth)

        # save output with iocontext
        fn_trained_model = self.ioctx.get_output_fn("trained_model.pickle")
        with open(fn_trained_model, "wb") as f:
            pickle.dump(model, f)

        self.logger.info("Model trained.")
