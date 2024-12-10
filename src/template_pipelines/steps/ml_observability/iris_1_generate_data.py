"""GenerateDataStep class."""
import pandas as pd
from sklearn import datasets

from aif.pyrogai.steps.step import Step


class GenerateDataStep(Step):
    """Generate Iris Dataset to provider."""

    def generate_dataframe(self) -> pd.DataFrame:
        """Loads Iris data and puts it in DataFrame."""
        data = datasets.load_iris()

        data_df = pd.DataFrame(data.data, columns=data.feature_names)
        data_df["class"] = data.target

        return data_df

    def run(self):
        """Runs step."""
        # Load data
        data_df = self.generate_dataframe()

        # Save output with iocontext
        fn = self.ioctx.get_output_fn("uploaded_data.parquet")
        data_df.to_parquet(fn)

        self.logger.info("Iris data generated.")
