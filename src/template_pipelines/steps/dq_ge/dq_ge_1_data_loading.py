"""Data loader step class."""
import random

import pandas as pd

from aif.pyrogai.steps.step import Step


class DataLoaderStep(Step):
    """Data loader step."""

    def generate_random_dataframe(self, num_rows=100, columns=None) -> pd.DataFrame:
        """Generate a random dataframe with specified columns and number of rows.

        This function can be deleted, it is only made for presentation purposes.
        """
        if columns is None:
            columns = ["column1", "column2", "column3"]

        data = []
        for _ in range(num_rows):
            row = [random.random() for _ in columns]
            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        return df

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Save the dataframe to ioctx."""
        fn = self.ioctx.get_output_fn(f"{filename}.csv")

        df.to_csv(fn)
        self.logger.info(f"Data saved to {filename}")

    def run(self):
        """Run."""
        self.logger.info("Start data loading step")

        # Generate and saving first dataset
        df1 = self.generate_random_dataframe(num_rows=150, columns=["A", "B", "C"])
        self.save_dataframe(df1, "data_set_1")

        # Generate and saving second dataset with different columns and row count
        df2 = self.generate_random_dataframe(num_rows=200, columns=["X", "Y"])
        self.save_dataframe(df2, "data_set_2")

        self.logger.info("Finish data loading step")
