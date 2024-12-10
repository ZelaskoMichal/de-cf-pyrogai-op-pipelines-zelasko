"""Model inference step class."""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.ml_inference.toolkit import AutoEncoder, convert_to_tensor

warnings.filterwarnings("ignore")


# Define ModelInference class and inherit properties from pyrogai Step class
class ModelInference(Step):
    """Model Inference step."""

    def calculate_mae(self, y, y_pred):
        """Calcuate mean absolute error."""
        return np.abs(y - y_pred).sum(1)

    def calculate_threshold(self, values, spread=0):
        """Calculate a threshold value based on mean and std."""
        return np.nanmean(values) + spread * np.nanstd(values)

    def load_model(self, n_features):
        """Load the trained anomaly model."""
        self.logger.info(f"Loading a trained model from path: {self.inputs['anomaly_model']}")
        if self.platform in ["AML", "Vertex", "Local"]:
            anomaly_model = AutoEncoder(n_features)
            anomaly_model.load_state_dict(torch.load(self.inputs["anomaly_model"]))
        elif self.platform in ["DBR"]:
            # For AML you can also use model URI if you want to
            anomaly_model = self.mlflow.pytorch.load_model(
                self.config["ml_inference"]["dbr_model_uri"]
            )
        anomaly_model.eval()
        self.logger.info("The trained model has been loaded.")
        return anomaly_model

    def detect_anomalies(self, dataset, thre):
        """Detect anomalies using a threshold value."""
        nans_mask = np.isnan(dataset["loss"])
        over_high_thr_mask = dataset["loss"] > thre
        self.logger.info(pd.Series(nans_mask).value_counts())
        self.logger.info(pd.Series(dataset["loss"] > thre).value_counts())
        full_mask = np.logical_or(nans_mask, over_high_thr_mask)
        self.logger.info(full_mask)
        self.logger.info(full_mask.sum())
        dataset["prediction"] = np.where(dataset["loss"].flat[full_mask], 1, 0)
        no_of_anomalies = np.count_nonzero(dataset["prediction"])

        self.logger.info(
            f"Prediction finished. Found {no_of_anomalies} anomalies "
            f"in {len(dataset['loss'])} rows."
        )
        return dataset

    def preprocess_data(self, training_data):
        """Preprocess and prepare data for prediction."""
        training_data = training_data.dropna(subset=[self.config["ml_inference"]["target"]])
        target = training_data[self.config["ml_inference"]["target"]]
        features = training_data.drop(columns=[self.config["ml_inference"]["target"]])

        preprocessor = joblib.load(self.inputs["impute_scaling_preprocessor"])
        features_preprocessed = pd.DataFrame(preprocessor.transform(features))

        dataset = {}
        dataset["data"] = features_preprocessed.dropna(axis=1, how="all")
        dataset["labels"] = abs(target - 1).astype(bool).reindex(dataset["data"].index)
        dataset["labels"].fillna(False, inplace=True)
        dataset["unredeemed"] = dataset["data"][dataset["labels"]]

        return dataset

    def visualize_distribution_of_losses(
        self, losses, low_thre, medium_thre, high_thre, output_dir
    ):
        """Visualize distribution of losses and its statiscal characteristics."""
        fig, axes = plt.subplots(nrows=1, sharex=True)
        axes.hist(losses, bins=30, edgecolor="black")
        axes.set_title("Reconstructions from model vs data")
        axes.axvline(x=low_thre, color="green", label="mean")
        axes.axvline(x=medium_thre, color="red", label="mean + std")
        axes.axvline(x=high_thre, color="orange", label="mean + 2.5std")
        axes.set_ylabel("# of examples")
        axes.legend(loc="upper right")
        plt.xlabel("Loss")
        plt.savefig(os.path.join(output_dir, "distribution_of_train_test_losses.png"))
        plt.close()
        self.mlflow.log_artifact(
            os.path.join(output_dir, "distribution_of_train_test_losses.png"),
            artifact_path="evaluation_plots",
        )

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model inference step."""
        self.logger.info("Running inference step...")

        # Download data
        input_path = self.ioctx.get_fn("feature_created/feature_created.parquet")
        training_data = pd.read_parquet(input_path)

        # Preprocess and get data in a format suitable for prediction
        dataset = self.preprocess_data(training_data)

        # Get and load the anomaly detector model trained in the ModelTraining step
        anomaly_model = self.load_model(len(dataset["unredeemed"].columns))

        # Create a directory to store model results (plots etc)
        output_dir = self.ioctx.get_output_fn("evaluated_model")
        os.makedirs(output_dir, exist_ok=True)

        # Convert to numpy
        data_np = dataset["unredeemed"].to_numpy()

        # Run prediction and get prediction errors
        reconstructions = anomaly_model(convert_to_tensor(data_np)).detach().numpy()
        loss = self.calculate_mae(reconstructions, data_np)
        dataset["loss"] = loss

        high_thre = self.calculate_threshold(dataset["loss"], 3)
        medium_thre = self.calculate_threshold(dataset["loss"], 1)
        low_thre = self.calculate_threshold(dataset["loss"])

        # Anomaly defined as either nan or over 3 standard deviations from mean
        dataset = self.detect_anomalies(dataset, high_thre)

        # Visualize distribution of errors/losses
        self.visualize_distribution_of_losses(
            dataset["loss"], low_thre, medium_thre, high_thre, output_dir
        )

        # Save output
        output_file = self.config["ml_inference"]["output_file"]
        pd.DataFrame(dataset["prediction"]).to_csv(f"{output_dir}/{output_file}")
        self.logger.info(f"File saved as {output_file}")
