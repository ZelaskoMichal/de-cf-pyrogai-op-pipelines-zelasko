"""Feature Creation step class."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step
from template_pipelines.utils.ml_training.toolkit import (
    left_join_all,
    load_tables,
    prefix_columns,
    q1,
    q3,
)


# Define FeatureCreation class and inherit properties from pyrogai Step class
class FeatureCreation(Step):
    """Feature Creation step."""

    def get_campaign_duration(self, tables):
        """Find a campaign duration."""
        tables["campaigns"]["campaign_duration"] = (
            tables["campaigns"]["end_date"] - tables["campaigns"]["start_date"]
        ).dt.days
        tables["campaigns"] = tables["campaigns"].drop(
            columns=[col for col in tables["campaigns"].columns if "date" in col]
        )

    def get_unit_price_and_categorical_variable(self, tables):
        """Find a unit price before discount."""
        # Create unit values from total values and a categorical variable from coupon_discount
        tables["customer_transactions"]["price_after_discount"] = (
            tables["customer_transactions"]["selling_price"]
            / tables["customer_transactions"]["quantity"]
        )
        tables["customer_transactions"]["discount"] = (
            tables["customer_transactions"]["other_discount"]
            / tables["customer_transactions"]["quantity"]
        )
        tables["customer_transactions"]["price_before_discount"] = (
            tables["customer_transactions"]["price_after_discount"]
            - tables["customer_transactions"]["discount"]
        )
        tables["customer_transactions"]["counpon_used"] = np.where(
            tables["customer_transactions"]["coupon_discount"] != 0, 1, 0
        )

    def get_customer_agg_trans(self, tables):
        """Aggreate values by customer_id to find summarized transactions on the customer level."""
        # Estimate appropriate metrics such as mode, sum, median, 25% and 75% quantiles
        # Add a prefix to columns to distinguish them from other later joined columns
        customer_agg_trans = (
            tables["customer_transactions"]
            .groupby("customer_id")
            .agg(
                no_of_unique_items=("item_id", lambda x: len(set(x))),
                no_of_items=("item_id", "sum"),
                median_quantity=("quantity", "median"),
                q1_quantity=("quantity", q1),
                q3_quantity=("quantity", q3),
                median_price=("price_before_discount", "median"),
                q1_price=("price_before_discount", q1),
                q3_price=("price_before_discount", q3),
                median_discount=("discount", "median"),
                q1_discount=("discount", q1),
                q3_discount=("discount", q3),
                median_coupon_discount=("coupon_discount", "median"),
                q1_coupon_discount=("coupon_discount", q1),
                q3_coupon_discount=("coupon_discount", q3),
                total_coupon_used=("counpon_used", "sum"),
                mode_dayofmonth=("dayofmonth", lambda x: pd.Series.mode(x)[0]),
                mode_dayofweek=("dayofweek", lambda x: pd.Series.mode(x)[0]),
                mode_month=("month", lambda x: pd.Series.mode(x)[0]),
            )
            .reset_index()
        )
        customer_agg_trans = prefix_columns(customer_agg_trans, "customer_", ["customer_id"])

        return customer_agg_trans

    def get_coupon_trans(self, tables):
        """Aggreate values by coupon_id to find summarized transactions on the coupon level."""
        # Estimate appropriate metrics such as mode, sum, median, 25% and 75% quantiles
        # Add a prefix to columns to distinguish them from other later joined columns
        coupon_item_trans = tables["coupon_items"].merge(
            tables["customer_transactions"], on="item_id", how="left"
        )
        coupon_trans = (
            coupon_item_trans.groupby("coupon_id")
            .agg(
                no_of_item=("item_id", lambda x: len(set(x))),
                mode_brand=("brand", lambda x: pd.Series.mode(x)[0]),
                mode_brand_type=("brand_type", lambda x: pd.Series.mode(x)[0]),
                mode_category=("category", lambda x: pd.Series.mode(x)[0]),
                no_of_customers=("customer_id", lambda x: len(set(x))),
                q1_quantity=("quantity", q1),
                q3_quantity=("quantity", q3),
                median_price=("price_before_discount", "median"),
                q1_price=("price_before_discount", q1),
                q3_price=("price_before_discount", q3),
                median_other_discount=("discount", "median"),
                q1_other_discount=("discount", q1),
                q3_other_discount=("discount", q3),
                median_coupon_discount=("coupon_discount", "median"),
                q1_coupon_discount=("coupon_discount", q1),
                q3_coupon_discount=("coupon_discount", q3),
                no_of_coupon_used=("counpon_used", "sum"),
            )
            .reset_index()
        )

        coupon_trans = prefix_columns(coupon_trans, "coupon_", exclude_columns=["coupon_id"])

        return coupon_trans

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Feature Creation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running feature creation...")

        # Get full path of file to read files
        # from some shared location using Pyrogai Step self.ioctx.get_fn
        # Read in a pickle file created in the Preprocessing step
        input_list = self.ioctx.get_fns("preprocessed/*.parquet")
        tables = load_tables(input_list, logger=self.logger)

        # Perform some feature engineering
        self.get_campaign_duration(tables)
        self.get_unit_price_and_categorical_variable(tables)
        customer_agg_trans = self.get_customer_agg_trans(tables)

        self.logger.info("Creating more features...")
        coupon_trans = self.get_coupon_trans(tables)

        # Left join all dataframes and create a training data
        # Keep columns without id
        join_tables = [
            (tables["coupon_redemption"], "coupon_id"),
            (coupon_trans, "coupon_id"),
            (tables["campaigns"], "campaign_id"),
            (tables["customer_demographics"], "customer_id"),
            (customer_agg_trans, "customer_id"),
        ]

        self.logger.info("Consolidating all features...")
        big_table = left_join_all(join_tables)
        column_names = [col for col in big_table.columns if "id" not in col]
        training_data = big_table[column_names]

        # Get full path of file to write training data as a parquet file
        # to some shared location using Pyrogai Step self.ioctx.get_output_fn
        output_dir = self.ioctx.get_output_fn("feature_created")
        os.makedirs(output_dir, exist_ok=True)
        training_data.to_parquet(os.path.join(output_dir, "feature_created.parquet"))

        # Create and store a heatmap of correlations for review
        correlations = training_data.corr()
        mask = np.triu(np.ones_like(correlations, dtype=bool))
        fig, ax = plt.subplots()
        sns.heatmap(correlations, mask=mask, cmap="BrBG")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), bbox_inches="tight")

        # For Vertex, use ioslot to visualize the heatmap in the Vertex AI pipelines UI
        if self.platform == Platform.VERTEX:
            self.outputs["kfp_md_plot"] = fig
        plt.close()

        # Log the heatmap of correlations in an mlflow artifact
        # MLFlow is fully integrated with AML and DBR, the run is set automatically
        self.mlflow.log_artifact(
            os.path.join(output_dir, "correlation_heatmap.png"),
            artifact_path="feature_creation_plots",
        )

        self.logger.info("Feature Creation is done.")
