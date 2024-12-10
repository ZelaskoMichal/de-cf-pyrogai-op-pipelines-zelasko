"""DataQualityStep class."""
import pandas as pd

from aif.pyrogai.steps.step import Step


class DataProcessingStep(Step):
    """Data processing step."""

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save the dataframe to ioctx."""
        fn = self.ioctx.get_output_fn(f"{filename}.csv")

        df.to_csv(fn)
        self.logger.info(f"Data saved to {filename}")

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform data processing on the generated dataframes."""
        # Standardizing data (example for df1)
        # Example: Normalize columns "A", "B", "C" to have mean 0 and standard deviation 1
        if all(col in df.columns for col in ["A", "B", "C"]):
            for col in ["A", "B", "C"]:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Scaling data (example for df2)
        # Example: Scale values in columns "X", "Y" to the range 0-1
        if all(col in df.columns for col in ["X", "Y"]):
            for col in ["X", "Y"]:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Adding a new column based on calculations (for both dfs)
        # Example: Add a column with the sum of values in each row
        df["sum"] = df.sum(axis=1)

        return df

    def run(self):
        """Runs step."""
        self.logger.info("Start data processing step")

        # Reading files from previous step
        df1 = pd.read_csv(self.ioctx.get_fn("data_set_1.csv"))
        df2 = pd.read_csv(self.ioctx.get_fn("data_set_2.csv"))

        processed_df1 = self.process_data(df1)
        processed_df2 = self.process_data(df2)

        self.save_dataframe(processed_df1, "processed_df1")
        self.save_dataframe(processed_df2, "processed_df2")

        self.logger.info(processed_df1)
        self.logger.info(processed_df2)

        self.logger.info("Finish data processing step")
