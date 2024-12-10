"""Prediction step class."""
import os

import matplotlib.pyplot as plt
import pandas as pd

from aif.pyrogai.steps.step import Step


class PredictionStep(Step):
    """Prediction step."""

    def run(self):
        """Runs step."""
        output_dir = self.ioctx.get_output_fn("graphs")

        min_model_result = self.mlflow.statsmodels.load_model(self.inputs["model_uri"])

        fn = self.ioctx.get_fn("co2_data.pkl")
        co2_series = pd.read_pickle(fn)

        self.save_forecast_graph(min_model_result, co2_series, output_dir)

    def save_forecast_graph(self, min_model, co2_series, output_dir):
        """Save forecast graph.

        Args:
            min_model (statsmodels.tsa.arima.model.ARIMA): The trained ARIMA model.
            co2_series (pandas.Series): The observed CO2 series.
            output_dir (str): The directory to save the forecast graph.

        Returns:
            None
        """
        pred_uc = min_model.get_forecast(steps=200)
        pred_ci = pred_uc.conf_int()
        ax = co2_series.plot(label="observed", figsize=(15, 8))
        pred_uc.predicted_mean.plot(ax=ax, label="Forecast")
        ax.fill_between(
            pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="k", alpha=0.25
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("CO2 Levels")

        plt.legend()
        os.makedirs(
            output_dir, exist_ok=True
        )  # Add this line to create the directory if it doesn't exist
        plt.savefig(os.path.join(output_dir, "forecast.png"))
        self.mlflow.log_artifact(
            os.path.join(output_dir, "forecast.png"),
            artifact_path="graphs",
        )
        plt.clf()
