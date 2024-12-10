"""Feature Creation step class."""

from aif.pyrogai.steps.step import Step


# Define FeatureCreation class and inherit properties from pyrogai Step class
class FeatureCreation(Step):
    """Feature Creation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Feature Creation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running feature creation...")

        # Add your feature creation here
        # Remember to properly factorize your code, adding modules
        # to your src folder as needed

        self.logger.info("Feature Creation is done.")
