"""Model evaluation step class."""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from aif.pyrogai.steps.step import Step


class ModelEvaluationStep(Step):
    """Model evaluation step."""

    def run(self):
        """Runs step."""
        output_dir = self.ioctx.get_output_fn("graphs")

        min_model_result = self.mlflow.statsmodels.load_model(self.inputs["model_uri"])
        pred = min_model_result.get_prediction(start=pd.to_datetime("1998-01-01"), dynamic=False)

        co2_forecasted = pred.predicted_mean

        fn = self.ioctx.get_fn("co2_data.pkl")
        co2_series = pd.read_pickle(fn)
        co2_truth = co2_series["1998-01-01":]

        self.generate_metrics(co2_truth, co2_forecasted)
        self.save_diagnostics_graph(min_model_result, output_dir)
        self.save_predict_vs_actual_graph(co2_series, pred, output_dir)

    def generate_metrics(self, co2_truth, co2_forecasted):
        """Compute metrics and save them to artifacts."""
        mse = mean_squared_error(co2_truth, co2_forecasted)
        self.mlflow.log_metric("Mean Squared Error", round(mse, 2))

        mae = mean_absolute_error(co2_truth, co2_forecasted)
        self.mlflow.log_metric("Mean Absolute Error", round(mae, 2))

        mape = mean_absolute_percentage_error(co2_truth, co2_forecasted)
        self.mlflow.log_metric("Mean Absolute Percentage Error", round(mape, 4))

        rmse = root_mean_squared_error(co2_truth, co2_forecasted)
        self.mlflow.log_metric("Root Mean Squared Error", round(rmse, 2))

    def save_diagnostics_graph(self, min_model, output_dir):
        """Save dianostics graph to artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        min_model.plot_diagnostics(figsize=(12, 8))
        plt.savefig(os.path.join(output_dir, "diagnostics.png"))
        self.mlflow.log_artifact(
            os.path.join(output_dir, "diagnostics.png"),
            artifact_path="graphs",
        )
        plt.clf()

    def save_predict_vs_actual_graph(self, co2_series, pred, output_dir):
        """Save forecast vs actual graph to artifacts."""
        ax = co2_series["1990":].plot(label="observed", figsize=(15, 6))
        pred.predicted_mean.plot(ax=ax, label="One-step-ahead forecast", alpha=0.7)
        pred_ci = pred.conf_int()

        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="k", alpha=0.2)

        ax.set_xlabel("Date")
        ax.set_ylabel("CO2 Levels")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "predict_vs_actual.png"))
        self.mlflow.log_artifact(
            os.path.join(output_dir, "predict_vs_actual.png"),
            artifact_path="graphs",
        )
        plt.clf()
