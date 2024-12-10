"""Model Evaluation step class."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (  # noqa: E402, E501
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.ml_training.toolkit import (  # noqa: E402
    convert_to_tensor,
    load_tables,
)

warnings.filterwarnings("ignore")


# Define ModelEvaluation class and inherit properties from pyrogai Step class
class ModelEvaluation(Step):
    """Model Evaluation step."""

    def calculate_mae(self, y, y_pred):
        """Calcuate mean absolute error."""
        return np.abs(y - y_pred).sum(1)

    def convert_labels_df_to_series(self, dataset):
        """Convert labels data frames to series."""
        dataset["test_labels"] = dataset["test_labels"].iloc[:, 0]
        dataset["train_labels"] = dataset["train_labels"].iloc[:, 0]

    def create_directory_to_store_model_evaluation(self):
        """Create a directory to store model evaluation results (plots etc)."""
        output_dir = self.ioctx.get_output_fn("evaluated_model")
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def visualize_feature_reconstructions_and_compare(
        self, dataset, n_features, output_dir, anomaly_detector
    ):
        """Visualize feature reconstructions and compare them with original features."""
        self.logger.info("Creating model performance stats for different threshold values...")
        unredeemed_decoded_data = (
            anomaly_detector(convert_to_tensor(dataset["test_unredeemed"].values)).detach().numpy()
        )
        redeemed_decoded_data = (
            anomaly_detector(convert_to_tensor(dataset["test_redeemed"].values)).detach().numpy()
        )

        data_points = [
            (dataset["test_unredeemed"], unredeemed_decoded_data, "Unredeemed"),
            (dataset["test_redeemed"], redeemed_decoded_data, "Redeemed"),
        ]
        fig, axes = plt.subplots(nrows=len(data_points), sharex=True)
        ix = 3
        for i in range(len(axes)):
            axes[i].plot(data_points[i][0].values[ix], "b")
            axes[i].plot(data_points[i][1][ix], "r")
            axes[i].fill_between(
                np.arange(n_features),
                data_points[i][1][ix],
                data_points[i][0].values[ix],
                color="lightcoral",
            )
            axes[i].set_title(data_points[i][2])
        plt.legend(
            labels=["Input", "Reconstruction", "Error"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            bbox_transform=fig.transFigure,
            ncol=3,
        )
        plt.savefig(os.path.join(output_dir, "feature_reconstructions.png"), bbox_inches="tight")
        plt.close()
        self.mlflow.log_artifact(
            os.path.join(output_dir, "feature_reconstructions.png"),
            artifact_path="evaluation_plots",
        )

    def visualize_distribution_of_losses(self, dataset, output_dir, anomaly_detector):
        """Visualize distribution of losses and its statiscal characteristics."""
        reconstructions = (
            anomaly_detector(convert_to_tensor(dataset["train_unredeemed"].values)).detach().numpy()
        )
        train_unredeemed_losses = self.calculate_mae(reconstructions, dataset["train_unredeemed"])

        reconstructions = (
            anomaly_detector(convert_to_tensor(dataset["test_redeemed"].values)).detach().numpy()
        )
        test_redeemed_losses = self.calculate_mae(reconstructions, dataset["test_redeemed"])

        reconstructions = (
            anomaly_detector(convert_to_tensor(dataset["test_data"].values)).detach().numpy()
        )
        test_losses = self.calculate_mae(reconstructions, dataset["test_data"])

        high_thre = np.mean(train_unredeemed_losses) + 3 * np.std(train_unredeemed_losses)
        medium_thre = np.mean(train_unredeemed_losses) + np.std(train_unredeemed_losses)
        low_thre = np.mean(train_unredeemed_losses)

        data_points = [
            (train_unredeemed_losses, "Train"),
            (test_redeemed_losses, "Test"),
        ]
        fig, axes = plt.subplots(nrows=len(data_points), sharex=True)
        for i in range(len(axes)):
            axes[i].hist(data_points[i][0], bins=30, edgecolor="black")
            axes[i].set_title(data_points[i][1])
            axes[i].axvline(x=low_thre, color="green", label="mean")
            axes[i].axvline(x=medium_thre, color="red", label="mean + std")
            axes[i].axvline(x=high_thre, color="orange", label="mean + 2.5std")
            axes[i].set_ylabel("# of examples")
        axes[0].legend(loc="upper right")
        plt.xlabel("Loss")
        plt.savefig(os.path.join(output_dir, "distribution_of_train_test_losses.png"))
        plt.close()
        self.mlflow.log_artifact(
            os.path.join(output_dir, "distribution_of_train_test_losses.png"),
            artifact_path="evaluation_plots",
        )

        return train_unredeemed_losses, test_losses

    def estimate_model_performace_stats(self, train_unredeemed_losses, test_losses, dataset):
        """Estimate model performace stats for different threshold values."""
        # Threshold values are calculated using formula:
        # mean + scalar * std where scalar is in [1,3)
        evaluation = []
        for scalar in np.arange(1, 3.0, 0.15):
            thre = np.mean(train_unredeemed_losses) + scalar * np.std(train_unredeemed_losses)
            results = np.less(test_losses, thre)
            cm = confusion_matrix(dataset["test_labels"], results, labels=[1, 0])
            tn, fp, fn, tp = cm.ravel()
            evaluation.append(
                (
                    scalar,
                    f"mean + {round(scalar, 2)} * std",
                    accuracy_score(dataset["test_labels"], results),
                    precision_score(dataset["test_labels"], results),
                    recall_score(dataset["test_labels"], results),
                    tn,
                    fp,
                    fn,
                    tp,
                )
            )
        evaluation = pd.DataFrame(
            evaluation,
            columns=[
                "std_scalar",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        )

        return evaluation

    def select_a_threshold_based(
        self, anomaly_detector, dataset, evaluation, train_unredeemed_losses
    ):
        """Select a threshold based on minimum true positive defined in config.json."""
        # Utilize pyrogai Step self.config
        min_tp = self.config["ml_training"]["min_tp"]
        self.logger.info(f"Selecting a threshold based on the minimum tp of {min_tp}...")
        reconstructions = (
            anomaly_detector(convert_to_tensor(dataset["train_data"].values)).detach().numpy()
        )
        train_losses = self.calculate_mae(reconstructions, dataset["train_data"])
        final_metrics = evaluation.loc[
            evaluation["tp"] >= min_tp, evaluation.columns != "threshold"
        ].iloc[-1]
        self.mlflow.log_metrics(final_metrics.to_dict())
        final_scalar = final_metrics["std_scalar"]
        final_threshold = np.mean(train_unredeemed_losses) + final_scalar * np.std(
            train_unredeemed_losses
        )
        self.mlflow.log_metric("final_threshold", final_threshold)
        self.logger.info("The threshold has been selected.")
        return train_losses, final_threshold, final_metrics

    def get_and_store_predictions(
        self, train_losses, final_threshold, dataset, output_dir, final_metrics
    ):
        """Get predictions based on the selected threshold."""
        # Create and store a confusion matrix visualization
        cm = final_metrics[["tn", "fp", "fn", "tp"]].to_numpy().reshape((2, 2))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Unredeemed", "Redeemed"])
        disp.plot(cmap="Blues", values_format="")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        self.mlflow.log_artifact(
            os.path.join(output_dir, "confusion_matrix.png"), artifact_path="evaluation_plots"
        )

        # For KFP, visualize all above saved plots together in Kubeflow UI
        if self.platform == Platform.VERTEX:
            images = []
            for image_filepath in output_dir.glob("*.png"):
                images.append(plt.imread(image_filepath))
            fig, axes = plt.subplots(len(images), 1, figsize=(15, 15))
            axes = axes.ravel()
            for i in range(len(images)):
                axes[i].imshow(images[i])
                axes[i].axis("off")
            plt.tight_layout()
            self.outputs["kfp_md_plot"] = fig
            plt.close()

        # Store data and predictions for the ModelObservability step
        self.logger.info("Preparing train data for model observability...")
        train_predictions = np.less(train_losses, final_threshold)
        train_data = dataset["train_data"].copy()
        train_data["actual"] = dataset["train_labels"].replace({True: 0, False: 1}).values
        train_data["prediction"] = ~train_predictions * 1
        train_data.to_parquet(os.path.join(output_dir, "mo_train_data.parquet"))
        self.logger.info("The train data for model observability have been prepared.")

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model evaluation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running model evaluation...")

        # Get full path of file to read data for model evaluation
        # using Pyrogai Step self.ioctx.get_fn
        input_path = self.ioctx.get_fns("imputed_scaled/*.parquet")
        dataset = load_tables(input_path)
        self.convert_labels_df_to_series(dataset)

        output_dir = self.create_directory_to_store_model_evaluation()
        n_features = len(dataset["train_unredeemed"].columns)

        # Get and load the anomaly detector model trained in the ModelTraining step
        self.logger.info(f"Loading a trained model from path: {self.inputs['model_uri']}")

        anomaly_detector = self.mlflow.pytorch.load_model(self.inputs["model_uri"])
        anomaly_detector.eval()
        self.logger.info("The trained model has been loaded.")

        self.visualize_feature_reconstructions_and_compare(
            dataset, n_features, output_dir, anomaly_detector
        )
        train_unredeemed_losses, test_losses = self.visualize_distribution_of_losses(
            dataset, output_dir, anomaly_detector
        )

        evaluation = self.estimate_model_performace_stats(
            train_unredeemed_losses, test_losses, dataset
        )

        # Log extra information using self.logger.info from Pyrogai Step class
        # Store model evaluation results
        self.logger.info(f"\n{evaluation}")
        evaluation.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"))
        self.logger.info("The model performance stats have been created.")

        # Select a threshold based on minimum true positive defined in config.json
        # Utilize pyrogai Step self.config
        train_losses, final_threshold, final_metrics = self.select_a_threshold_based(
            anomaly_detector, dataset, evaluation, train_unredeemed_losses
        )

        # Get predictions based on the selected threshold
        # Store data and predictions for the ModelObservability step
        self.get_and_store_predictions(
            train_losses, final_threshold, dataset, output_dir, final_metrics
        )
        self.logger.info("Model evaluation is done.")
