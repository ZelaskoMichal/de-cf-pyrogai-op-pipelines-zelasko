"""Preprocessing step class."""

import warnings
from pathlib import Path

import db_dtypes  # noqa: F401
import pandas as pd

from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.ml_training.toolkit import label_encode, save_tables  # noqa: E402

warnings.filterwarnings("ignore")


# Define Preprocessing class and inherit properties from pyrogai Step class
class Preprocessing(Step):
    """Preprocessing step."""

    def tables_into_dict(self):
        """Put tables in a dictionary for easier pre-processing."""
        # Read in data using input ioslots
        tables = {}
        for name, _ in self.inputs.items():
            # Get basename without extension
            basename = Path(name).stem
            tables[basename] = pd.read_parquet(self.inputs[name])
            self.logger.info(
                f"Checking {name} data for preprocessing...\n{tables[basename].head(2)}"
            )
        return tables

    def transform_coupon_redemption(self, tables):
        """Transforms 'coupon_redemption' table."""
        tables["coupon_redemption"] = tables["coupon_redemption"].dropna(
            subset=["redemption_status"]
        )

    def transform_customer_transactions(self, tables):
        """Transforms 'customer_transactions' table."""
        tables["customer_transactions"]["date"] = pd.to_datetime(
            tables["customer_transactions"]["date"]
        )
        tables["customer_transactions"] = tables["customer_transactions"].sort_values(
            by=["date"], ascending=True, ignore_index=True
        )

    def transform_campaigns(self, tables):
        """Transforms 'campaigns' table."""
        for col in tables["campaigns"].columns:
            if "date" in col:
                tables["campaigns"][col] = pd.to_datetime(tables["campaigns"][col])

    def rename_and_generalize_columns(self, tables):
        """Renames and generalizes 'customer_demographics' table columns."""
        tables["customer_demographics"] = tables["customer_demographics"].rename(
            columns={"marital_status": "relationship_status"}
        )
        tables["customer_demographics"]["relationship_status"] = tables["customer_demographics"][
            "relationship_status"
        ].apply(lambda x: "In relationship" if x == "Married" else x)

    def fill_missing_values(self, tables):
        """Fill in missing values with some assumptions."""
        # Assume it is a nuclear family
        # If family_size==1, then relationship_status='Single'
        # If family_size==2 and no_of_children is missing,
        # then relationship_status='In relationship'
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["relationship_status"].isnull())
            & (tables["customer_demographics"]["family_size"] == "1"),
            "relationship_status",
        ] = "Single"
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["relationship_status"].isnull())
            & (tables["customer_demographics"]["no_of_children"].isnull())
            & (tables["customer_demographics"]["family_size"] == "2"),
            "relationship_status",
        ] = "In relationship"

    def create_additional_columns(self, tables):
        """Create additional columns with special characters remove from strings."""
        tables["customer_demographics"]["family_size_int"] = tables["customer_demographics"][
            "family_size"
        ].apply(lambda x: float(x.replace("+", "")))

        tables["customer_demographics"]["no_of_children_int"] = tables["customer_demographics"][
            "no_of_children"
        ].apply(lambda x: float(x.replace("+", "")) if pd.notna(x) else x)

    def single_or_in_relationship(self, tables):
        """**Set single or in relationship status**.

        If family_size - no_of_children == 1, then relationship_status='Single'
        If family_size - no_of_children == 2, then relationship_status='In relationship'.
        """
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["relationship_status"].isnull())
            & (~tables["customer_demographics"]["family_size"].str.contains("\+"))  # noqa: W605
            & (
                (
                    tables["customer_demographics"]["family_size_int"]
                    - tables["customer_demographics"]["no_of_children_int"]
                )
                == 1
            ),
            "relationship_status",
        ] = "Single"
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["relationship_status"].isnull())
            & (~tables["customer_demographics"]["family_size"].str.contains("\+"))  # noqa: W605
            & (
                ~tables["customer_demographics"]["no_of_children"]
                .astype(str)
                .str.contains("\+")  # noqa: W605
            )
            & (
                (
                    tables["customer_demographics"]["family_size_int"]
                    - tables["customer_demographics"]["no_of_children_int"]
                )
                == 2
            ),
            "relationship_status",
        ] = "In relationship"

    def drop_unneeded_columns(self, tables):
        """Drop unneeded columns for further preprocessing."""
        tables["customer_demographics"] = tables["customer_demographics"].drop(
            columns=[col for col in tables["customer_demographics"].columns if "int" in col]
        )

    def set_no_of_children(self, tables):
        """**Setting no of children**.

        If relationship_status=='In relationship' and family_size==2, then no_of_children=0
        If family_size==1, then no_of_children=0
        If relationship_status=='Single' and family_size==2, then no_of_children=1
        """
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["no_of_children"].isnull())
            & (tables["customer_demographics"]["relationship_status"] == "In relationship")
            & (tables["customer_demographics"]["family_size"] == "2"),
            "no_of_children",
        ] = "0"
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["no_of_children"].isnull())
            & (tables["customer_demographics"]["family_size"] == "1"),
            "no_of_children",
        ] = "0"
        tables["customer_demographics"].loc[
            (tables["customer_demographics"]["no_of_children"].isnull())
            & (tables["customer_demographics"]["relationship_status"] == "Single")
            & (tables["customer_demographics"]["family_size"] == "2"),
            "no_of_children",
        ] = "1"

    def transform_customer_demographics(self, tables):
        """Transforms 'customer_demographics' table."""
        self.rename_and_generalize_columns(tables)
        self.fill_missing_values(tables)
        self.create_additional_columns(tables)
        self.single_or_in_relationship(tables)
        self.drop_unneeded_columns(tables)
        self.set_no_of_children(tables)

    def data_transformation(self, tables):
        """Data transformation."""
        self.logger.info("Running data transformation...")

        self.transform_coupon_redemption(tables)
        self.transform_customer_transactions(tables)
        self.transform_campaigns(tables)
        self.transform_customer_demographics(tables)

        self.logger.info("Data have been transformed.")

    def label_encoding(self, tables):
        """Label encode categorical variables."""
        self.logger.info("Running label encoding...")
        tables["campaigns"], _ = label_encode(tables["campaigns"], ["campaign_type"])
        tables["items"], _ = label_encode(tables["items"], ["brand_type", "category"])
        coupon_items = tables["coupon_item_mapping"].merge(
            tables["items"], on="item_id", how="left"
        )
        tables["coupon_items"] = coupon_items
        customer_columns = {
            "categorical": [
                "age_range",
                "relationship_status",
                "family_size",
                "no_of_children",
            ]
        }
        tables["customer_demographics"], _ = label_encode(
            tables["customer_demographics"], customer_columns["categorical"]
        )

        tables["customer_demographics"]["customer_id"] = tables["customer_demographics"][
            "customer_id"
        ].astype(int)

        # Label encode dates
        tables["customer_transactions"]["dayofmonth"] = tables["customer_transactions"][
            "date"
        ].dt.day
        tables["customer_transactions"]["dayofweek"] = tables["customer_transactions"][
            "date"
        ].dt.weekday
        tables["customer_transactions"]["month"] = tables["customer_transactions"]["date"].dt.month
        self.logger.info("Data have been label encoded.")

    # Pyrogai executes code defined under run method
    def run(self):
        """Run preprocessing step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running preprocessing...")
        # Check platform provider
        self.logger.info(f"The pipeline is running on {self.platform} platform.")

        # Read in data and perform some data transformation
        tables = self.tables_into_dict()
        self.data_transformation(tables=tables)
        self.label_encoding(tables)

        # Save preprocessed files to shared work directory using Pyrogai Step self.ioctx.get_output_fn
        output_dir = self.ioctx.get_output_fn("preprocessed")
        save_tables(tables, output_dir, self.logger)

        self.logger.info("Preprocessing is done.")
