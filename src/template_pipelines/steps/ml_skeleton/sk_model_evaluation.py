"""Model Evaluation step class."""

# Define necessary imports like os or warnings

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define ModelEvaluation class and inherit properties from pyrogai Step class
class ModelEvaluation(Step):
    """Model Evaluation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model evaluation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running model evaluation...")

        # load the saved model and log some evaluation metrics and artifacts

        self.logger.info("Model evaluation is done.")
