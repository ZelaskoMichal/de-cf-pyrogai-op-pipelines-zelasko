"""Preprocessing step class."""

import pandas as pd

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define Preprocessing class and inherit properties from pyrogai Step class
class Preprocessing(Step):
    """Preprocessing step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run preprocessing step."""
        # Log information using self.logger.info from Pyrogai Step class
        # Get full path of file to write
        # to some shared location using Pyrogai Step self.ioctx.get_output_fn
        self.logger.info("Running preprocessing...")

        # This example illustrates how to read data from input slots.
        # You simply use the self.inputs dictionary to get a file name
        # you can use as if this was a local file.
        # The same code will work without any modifications on the cloud.
        df = pd.read_parquet(self.inputs["input_data"])

        # Now we save the file to "IO Context".
        # This is a shared location that is accessible by all steps.
        # On the cloud, each step runs as a separate container.
        # This capability makes it easy to share data between steps
        # without having to copy it around or worrying about complexity.
        output_file = self.ioctx.get_output_fn("data.parquet")
        df.to_parquet(output_file)

        self.logger.info("Preprocessing is done.")
