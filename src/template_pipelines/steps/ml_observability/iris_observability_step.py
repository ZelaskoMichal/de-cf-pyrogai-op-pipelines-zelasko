"""Model Observability step class."""

import pandas as pd

from aif.pyrogai.steps.observability_step import ObservabilityStep


class ModelObservability(ObservabilityStep):
    """Model Observability step."""

    def run(self):
        """Run model observability step."""
        self.logger.info("Sending model training data...")

        target = self.config["ml_observability"]["target"]
        features = self.config["ml_observability"]["features"]
        predicted = f"{self.config['ml_observability']['target']}_predicted"

        # Upload train data
        train_data_path = self.ioctx.get_fn("train_predicted.parquet")
        train_data = pd.read_parquet(train_data_path)

        self.observability_clients["Iris_Classifier"].prepare_data(
            data=train_data.reset_index(),
            model_input_features=features,
            actual_column=target,
            output_column=predicted,
            unique_key_columns=["index"],
        )

        self.observability_clients["Iris_Classifier"].send_data("train")
        self.logger.info("Model training data uploaded.")

        # Upload test data
        test_data_path = self.ioctx.get_fn("test_predicted.parquet")
        test_data = pd.read_parquet(test_data_path)

        self.observability_clients["Iris_Classifier"].prepare_data(
            data=test_data.reset_index(),
            model_input_features=features,
            actual_column=target,
            output_column=predicted,
            unique_key_columns=["index"],
            batch_id="1",
        )

        self.observability_clients["Iris_Classifier"].send_data(
            "test",
        )
        self.logger.info("Model test data uploaded.")
