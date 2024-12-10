"""Imputation and Scaling class."""

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define ImputationScaling class and inherit properties from pyrogai Step class
class ImputationScaling(Step):
    """Imputation and Scaling step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run imputation and scaling step."""
        # Set seed to replicate the results
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running imputation...")

        # Many machine learning methods run better with properly normalized features
        # Use this step to normalize your features and impute any missing values

        self.logger.info("Imputation and scaling is done.")
