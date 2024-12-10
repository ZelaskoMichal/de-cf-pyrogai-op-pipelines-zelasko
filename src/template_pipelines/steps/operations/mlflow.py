"""Mlflow step class."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from aif.pyrogai.steps.step import Step  # noqa


class MlflowStep(Step):
    """MLflow step."""

    def run(self):
        """Run mlflow step."""
        # Unfortunately, for now mlflow is not supported on VertexAI yet
        # for Vertex use "vertex_meta" pipeline from TPT repo
        if self.platform.lower() != "vertex":
            # Loading data about California"s housing market
            data = fetch_california_housing()
            x = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.DataFrame(
                data.target, columns=["MedHouseVal"]
            )  # MedHouseVal is the median house value

            # Splitting data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            # mlflow params logging
            self.mlflow.log_param("test_size", 0.2)
            self.mlflow.log_param("random_state", 42)

            # Training linear regression
            model = LinearRegression()
            model.fit(x_train, y_train.values.ravel())

            # Model evaluation
            y_pred = model.predict(x_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            # Save metrics etc. into pipeline
            self.mlflow_utils.log(log_to_root_run=True, metrics={"RMSE": rmse}, params={}, tags={})

            # Saving plot (artifacts) with example of creating temporary file and ioslots
            with NamedTemporaryFile(suffix=".png") as tf:
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted Values")
                plt.plot(
                    [min(y_test.values), max(y_test.values)],
                    [min(y_test.values), max(y_test.values)],
                    color="red",
                )
                plt.grid(True)
                plt.savefig(tf.name)
                plt.close()

                # saving image to mlflow artifacts
                self.outputs["RMSE_california_flats"] = Path(tf.name)

            """
            Why something is saved into params not into metric?
            |
            The main difference is that parameters are fixed
            and define the experiment"s configuration, while metrics are
            variables that change over time and are used to evaluate the experiment"s results.
            MLflow efficiently tracks both types of data,
            facilitating the analysis and comparison of different experiments and models.
            |
            """
