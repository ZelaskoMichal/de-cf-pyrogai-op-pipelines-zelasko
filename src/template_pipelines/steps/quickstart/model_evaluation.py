"""Model Evaluation step class."""

# Define necessary imports like os or warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define ModelEvaluation class and inherit properties from pyrogai Step class
class ModelEvaluation(Step):
    """Model Evaluation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model evaluation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running model evaluation...")

        # Load test data
        x_test = pd.read_parquet(self.ioctx.get_fn("train_test_data/x_test.parquet"))
        y_test = pd.read_parquet(self.ioctx.get_fn("train_test_data/y_test.parquet"))

        # Load Model
        clf = self.mlflow.sklearn.load_model(self.inputs["model_uri"])

        # Score dataset
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)

        # compute and log metrics
        self.mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        self.mlflow.log_metric("precision", precision_score(y_test, y_pred))
        self.mlflow.log_metric("recall", recall_score(y_test, y_pred))
        self.mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        self.mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_prob[:, 1]))

        # We want to use accuracy as the metrics to compare different model runs,
        # so we also log it to the parent run for easier comparison
        self.mlflow_utils.log(
            log_to_root_run=True,
            metrics={"accuracy": accuracy_score(y_test, y_pred)},
        )

        # Plot and log ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])

        fig_roc = plt.figure()

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        fig_roc.savefig(self.ioctx.get_output_fn("roc_curve.png"))
        self.mlflow.log_artifact(self.ioctx.get_output_fn("roc_curve.png"))

        # Plot and log feature importance
        features = x_test.columns

        importances = clf.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [features[i] for i in indices]

        # Create plot
        plt.figure()
        plt.title("Feature Importance")
        plt.bar(range(x_test.shape[1]), importances[indices])
        plt.xticks(range(x_test.shape[1]), names, rotation=90)

        plt.tight_layout()

        plt.savefig(self.ioctx.get_output_fn("feature_importance.png"))
        self.mlflow.log_artifact(self.ioctx.get_output_fn("feature_importance.png"))

        # load the saved model and log some evaluation metrics and artifacts
        self.logger.info("Model evaluation is done.")
