"""Preprocessing step class."""

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

        # in this step you can do some basic data preprocessing and checks
        # (e.g. adjust type and log basic data statistics)
        # And you can copy the data to the ioctx using self.ioctx (see docs)

        self.logger.info("Preprocessing is done.")
