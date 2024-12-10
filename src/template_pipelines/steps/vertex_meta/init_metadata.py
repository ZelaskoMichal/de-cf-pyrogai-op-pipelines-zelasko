"""Init Metadata step class."""
import os
import secrets

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import vertexai
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step


class InitMetadata(Step):
    """Data loader step."""

    def __init__(self, *args, **kwargs):
        """Custom step initialization."""
        super().__init__(*args, **kwargs)
        vertexai.init(
            project=self.config["vertex_meta"]["project_name"],
            experiment=self.config["vertex_meta"]["experiment"],
            staging_bucket=self.config["vertex_meta"]["staging_bucket"],
        )

    @staticmethod
    def generate_random_dataframe(num_rows=100, columns=None) -> pd.DataFrame:
        """Generate a random dataframe with specified columns and number of rows.

        This function can be deleted, it is only made for presentation purposes.
        """
        if columns is None:
            columns = ["column1", "column2", "column3"]

        data = []
        for _ in range(num_rows):
            row = [secrets.randbelow(1000) for _ in columns]
            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        return df

    @staticmethod
    def _prepare_data(df):
        x = df[["column2", "column3"]]
        y = df["column1"]
        return train_test_split(x, y, random_state=0)

    @staticmethod
    def _train_model(x_train, y_train):
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def _evaluate_model(model, x_test, y_test):
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)
        metrics = {"mse": mse, "mae": mae, "r_squared": r_squared}
        params = {f"coeff_{i}": coeff for i, coeff in enumerate(model.coef_)}
        params["intercept"] = model.intercept_
        return metrics, params, y_pred

    def _save_model(self, model):
        output_dir = self.ioctx.get_output_fn("model")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "linear_regression_model.joblib")
        joblib.dump(model, model_path)

    def run(self):
        """Running code logic."""
        df = self.generate_random_dataframe()
        output_dir = self.ioctx.get_output_fn("dataframe")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "test_df.csv"))

        x_train, x_test, y_train, y_test = self._prepare_data(df)

        model = self._train_model(x_train, y_train)

        metrics, params, y_pred = self._evaluate_model(model, x_test, y_test)

        self.logger.info(model.coef_)
        self.logger.info(model.intercept_)
        self.logger.info(type(params))
        self.logger.info(params)

        self._plot_preds(y_test, y_pred)

        self._save_model(model)
        self.outputs["metrics"] = metrics
        self.outputs["params"] = params

    def _plot_preds(self, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        output_dir = self.ioctx.get_output_fn("plot")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "test_artifact_plotv1.png"))
        if self.platform == Platform.VERTEX:
            self.outputs["artifact"] = fig
