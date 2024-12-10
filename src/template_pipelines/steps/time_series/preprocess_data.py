"""Preprocess step class."""

import pandas as pd
import statsmodels.api as sm

from aif.pyrogai.steps.step import Step


class PreprocessDataStep(Step):
    """Read time series data."""

    def generate_data(self) -> pd.Series:
        """Generate data function."""
        data = sm.datasets.co2.load_pandas()
        return data.data["co2"]

    def adjust_data(self, co2_df: pd.Series) -> pd.Series:
        """Adjust data function."""
        co2_df = co2_df.resample("MS").mean()
        co2_df = co2_df.fillna(co2_df.bfill())
        return co2_df

    def run(self):
        """Runs step."""
        co2_df = self.generate_data()
        co2_df = self.adjust_data(co2_df=co2_df)

        fn = self.ioctx.get_output_fn("co2_data.pkl")
        co2_df.to_pickle(fn)
