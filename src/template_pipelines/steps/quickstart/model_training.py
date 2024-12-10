"""Model training step class."""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define ModelTraining class and inherit properties from pyrogai Step class
class ModelTraining(Step):
    """Model Training step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model training step."""
        # Set seed to replicate the results

        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Building and compiling a model...")

        # Now we can read the data from IO Context, in a very similar way.
        # We just need to use the get_input_fn method to get a file name.
        file_name = self.ioctx.get_fn("data.parquet")
        df = pd.read_parquet(file_name)

        # We get the model parameters from the configuration file,
        # and log them to mlflow.
        features = self.config["quickstart"]["features"]
        n_estimators = self.config["quickstart"]["n_estimators"]
        random_state = self.config["quickstart"]["random_state"]

        self.mlflow.log_param("features", features)
        self.mlflow.log_param("n_estimators", n_estimators)
        self.mlflow.log_param("random_state", random_state)

        # Now we can train the model.

        # Assuming you have a DataFrame `df` and your target column is 'target'
        X = df.drop("target", axis=1)[features]  # Features
        y = df["target"]  # Target

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Create a Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

        # Fit the classifier with the training data
        clf.fit(x_train, y_train)

        # Log the model to mlflow
        mlinfo = self.mlflow.sklearn.log_model(clf, "classifier")

        # Pass the model URI to an output slot
        # So that it can be retrieved by the next step and also outside of the pipeline
        self.outputs["model_uri"] = mlinfo.model_uri

        # Persist the data sets to IO Context
        # This time we can create a folder to store multiple files
        # (pandas series cannot be saved to parquet, so we convert them to dataframes first)
        output_folder = self.ioctx.get_output_fn("train_test_data")
        os.makedirs(output_folder, exist_ok=True)

        x_train.to_parquet(output_folder / "x_train.parquet")
        x_test.to_parquet(output_folder / "x_test.parquet")

        y_train.to_frame().to_parquet(output_folder / "y_train.parquet")
        y_test.to_frame().to_parquet(output_folder / "y_test.parquet")

        self.logger.info(f"The model has been trained and saved")
