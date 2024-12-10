"""Video Result Aggregation step class."""

from tempfile import NamedTemporaryFile

import pandas as pd

from aif.pyrogai.steps.step import Step


# Define ResultAggregation class and inherit properties from pyrogai Step class
class ResultAggregation(Step):
    """Video Result Aggregation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Video Result Aggregation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running video result aggregation...")

        # Get a full filepath to read a file created in the Video Processing step
        # and stored in a shared location using Pyrogai Step self.ioctx.get_fn
        input_path = self.ioctx.get_fn("video_processing/responses.txt")
        with open(input_path, "r") as f:
            lines = f.readlines()

        # Merge responses from the Gemini model with the original data
        rows = [line.strip().split(";") for line in lines]
        original_df = pd.read_csv(self.inputs["video_data.csv"], sep=";", dtype=str)
        processed_df = pd.DataFrame(
            rows, columns=["id", "product", "usage", "pros", "cons", "recommendation"]
        ).astype(str)
        video_results = original_df.merge(processed_df, on="id", how="left")

        # Write data into ioslot-defined output
        with NamedTemporaryFile() as f:
            video_results.to_csv(f.name, sep=";")
            self.outputs["video_results.csv"] = f.name

        self.logger.info("Video result aggregation is done.")
