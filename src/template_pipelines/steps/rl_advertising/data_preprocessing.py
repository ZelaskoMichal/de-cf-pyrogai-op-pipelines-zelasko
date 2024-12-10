"""Preprocessing step class."""

import numpy as np
import pandas as pd

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.rl_advertising.toolkit import save_tables


# Define Preprocessing class and inherit properties from pyrogai Step class
class Preprocessing(Step):
    """Preprocessing step."""

    def transform_categorical(self, df):
        """Remove trailing/leading white spaces and convert characters to lowercase."""
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].apply(lambda x: x.strip().lower())

        return df

    def reduce_categories(self, df):
        """Reduce number of categories for specified categorical variables."""
        edu_mapping = {
            "preschool": "elementary",
            "1st-4th": "elementary",
            "5th-6th": "elementary",
            "7th-8th": "elementary",
            "9th": "middle",
            "10th": "middle",
            "11th": "middle",
            "12th": "middle",
            "some-college": "undergraduate",
            "bachelors": "undergraduate",
            "assoc-acdm": "undergraduate",
            "assoc-voc": "undergraduate",
            "prof-school": "graduate",
            "masters": "graduate",
            "doctorate": "graduate",
        }
        df["education"] = df["education"].replace(edu_mapping)
        df["native_country"] = df["native_country"].replace({"united-states": "us"})
        df.loc[df["native_country"] != "us", "native_country"] = "non-us"
        df["workclass"] = df["workclass"].replace(["never-worked", "without-pay", "?"], "undefined")

        return df

    def create_categories(self, df):
        """Create new categories for specified categorical variables."""
        df.loc[
            (df["occupation"] == "?") | (df["occupation"] == "other-service"), "occupation"
        ] = "other"
        df["occupation"] = df.apply(
            lambda x: (
                x["occupation"] + "-2"
                if "graduate" in x["education"] and x["occupation"] == "other"
                else x["occupation"]
            ),
            axis=1,
        )

        return df

    def one_hot_encode(self, df, except_column):
        """One-hot encode categorical variables except for a specified one."""
        context_cols = [c for c in df.columns if c != except_column]
        df = pd.concat([pd.get_dummies(df[context_cols]).astype(float), df[except_column]], axis=1)

        return df

    def split_data(self, df, frac):
        """Split dataframe into two dataframes based on fraction."""
        train_df = df.sample(frac=frac)
        test_df = df.drop(train_df.index)

        return {"train_df": train_df, "test_df": test_df}

    # Pyrogai executes code defined under run method
    def run(self):
        """Run preprocessing step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running data preprocessing...")

        # Read in ioslot-defined data
        df = pd.read_parquet(self.inputs["us_census_data.parquet"])
        df = df.drop(df.filter(regex="capital|num|fnlwgt").columns, axis=1)
        df = self.transform_categorical(df)

        # Reduce number of unique categories
        df = self.reduce_categories(df)

        # Create new categories
        df = self.create_categories(df)

        # Drop missing values
        df = df.replace("?", np.nan).dropna()

        # One-hot encode categorical variables except for education
        df = self.one_hot_encode(df, except_column="education")

        # Split data into training and test sets
        tables = self.split_data(df, frac=self.config["rl_advertising"]["train_size"])

        # Save preprocessed data to a shared work directory using ioctx
        output_dir = self.ioctx.get_output_fn("data_preprocessing")
        save_tables(tables, output_dir, self.logger)

        self.logger.info("Preprocessing is done.")
