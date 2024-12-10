"""Imputation and Scaling class."""

import os
import random
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.ml_training.toolkit import save_tables  # noqa: E402

warnings.filterwarnings("ignore")


# Define ImputationScaling class and inherit properties from pyrogai Step class
class ImputationScaling(Step):
    """Imputation and Scaling step."""

    def get_preprocessor_pipeline(self, features, x_train, y_train):
        """Get pipeline for features and target from training data."""
        self.logger.info("Start defining imputation and scaling pipeline")
        categorical_columns = [
            "coupon_mode_brand",
            "coupon_mode_brand_type",
            "coupon_mode_category",
            "campaign_type",
            "age_range",
            "relationship_status",
            "rented",
            "family_size",
            "no_of_children",
            "income_bracket",
            "customer_mode_dayofmonth",
            "customer_mode_dayofweek",
            "customer_mode_month",
        ]
        # Features (no target) without categorical should yield us all numerical columns.
        numeric_columns = list(set(features.columns.values.tolist()) - set(categorical_columns))
        numeric_transformer = Pipeline(
            steps=[("scaler", MinMaxScaler()), ("imputer", SimpleImputer(strategy="mean"))]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean", fill_value="unknown")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ]
        )
        self.logger.info("Run impute-scaling preprocessing on features set.")
        preprocessor.fit(x_train, y_train)
        return preprocessor

    def transform_and_save(self, x_train, x_test, preprocessor):
        """Transform data to created pipeline preprocessor and save it as file."""
        x_train_scaled = pd.DataFrame(preprocessor.transform(x_train))
        x_test_scaled = pd.DataFrame(preprocessor.transform(x_test))

        output_dir = self.ioctx.get_output_fn("impute_scaling")
        preprocessor_file = os.path.join(output_dir, "impute_scaling_preprocessor.pkl")
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_file)
        self.outputs["impute_scaling_preprocessor"] = preprocessor_file

        return x_train_scaled, x_test_scaled

    def split_data_into_unredeemed_and_redeemed(
        self, y_train, y_test, x_train_scaled, x_test_scaled
    ):
        """Create a dictionary of different dataframes for later usage."""
        # Divide data into unredeemed and redeemed coupon datasets
        # Treat unredeemed as normal (positive) whereas redeemed as anomolous (negative)

        features_scaled = pd.concat([x_train_scaled, x_test_scaled])

        dataset = {}
        dataset["train_data"] = x_train_scaled.dropna(axis=1, how="all")
        dataset["test_data"] = x_test_scaled.dropna(axis=1, how="all")

        dataset["train_labels"] = abs(y_train - 1).astype(bool).reindex(dataset["train_data"].index)
        dataset["test_labels"] = (
            abs(y_test - 1).astype(bool).fillna(False).reindex(dataset["test_data"].index)
        )
        dataset["train_labels"].fillna(False, inplace=True)
        dataset["test_labels"].fillna(False, inplace=True)

        dataset["train_unredeemed"] = dataset["train_data"][dataset["train_labels"]]
        dataset["test_unredeemed"] = dataset["test_data"].loc[dataset["test_labels"], :]

        dataset["train_redeemed"] = dataset["train_data"].loc[~dataset["train_labels"], :]
        dataset["test_redeemed"] = dataset["test_data"].loc[~dataset["test_labels"], :]

        # Convert to DF for the purpose of saving
        dataset["train_labels"] = pd.DataFrame(abs(y_train - 1).astype(bool))
        dataset["test_labels"] = pd.DataFrame(abs(y_test - 1).astype(bool))

        dataset["imputed_scaled"] = features_scaled

        # Get full path of file to write imputed and scaled data
        # to some shared location using Pyrogai Step self.ioctx.get_output_fn
        output_dir = self.ioctx.get_output_fn("imputed_scaled")
        save_tables(dataset, output_dir, self.logger)

        # Create and store ECG visualization
        # to show behavioral patterns of normal and anomalous features
        data_points = [
            (dataset["train_unredeemed"], "Normal ECG (Coupon Unredeemed)"),
            (dataset["train_redeemed"], "Anomalous ECG (Coupon Redeemed)"),
        ]
        fig, axes = plt.subplots(nrows=len(data_points), sharex=True)
        ix = 3
        for i in range(len(axes)):
            axes[i].plot(np.arange(len(data_points[i][0].columns)), data_points[i][0].iloc[ix])
            axes[i].set_title(data_points[i][1])
        plt.savefig(os.path.join(output_dir, "ECG.png"))

        # For Vertex, use ioslot to visualize the ECG in the Vertex AI pipelines UI
        if self.platform == Platform.VERTEX:
            self.outputs["kfp_md_plot"] = fig
        plt.close()

        # Log the ECG visualization in an mlflow artifact
        # MLFlow is fully integrated with AML and DBR, the run is set automatically
        self.mlflow.log_artifact(
            os.path.join(output_dir, "ECG.png"),
            artifact_path="imputation_scaling_plots",
        )

    # Pyrogai executes code defined under run method
    def run(self):
        """Run imputation and scaling step."""
        # Set seed to replicate the results
        seed = self.config["ml_training"]["random_state"]
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Get full path of file to read a parquet file
        # from some shared location using Pyrogai Step self.ioctx.get_fn
        # Read in training data created in the FeatureCreation step
        input_path = self.ioctx.get_fn("feature_created/feature_created.parquet")
        training_data = pd.read_parquet(input_path)
        training_data = training_data.dropna(subset=[self.config["ml_training"]["target"]])

        # Utilize Step self.config to get the target defined in config.json as well as the features
        # Split the data into training and test sets
        target = training_data[self.config["ml_training"]["target"]]
        features = training_data.drop(columns=[self.config["ml_training"]["target"]])
        self.logger.info("Splitting data into training and test sets...")
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=seed
        )

        self.logger.info("Running pipeline transform...")
        preprocessor = self.get_preprocessor_pipeline(features, x_train, y_train)
        self.logger.info("Pipeline transform is done.")

        self.logger.info("Transforming data using preprocessor model...")
        x_train_scaled, x_test_scaled = self.transform_and_save(x_train, x_test, preprocessor)
        self.logger.info("Transformation is done.")

        self.logger.info("Splitting data into training and test sets...")
        self.split_data_into_unredeemed_and_redeemed(y_train, y_test, x_train_scaled, x_test_scaled)
        self.logger.info("Splitting data is done.")
