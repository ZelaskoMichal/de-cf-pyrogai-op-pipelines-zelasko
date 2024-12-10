"""Model training step class."""

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define ModelTraining class and inherit properties from pyrogai Step class
class ModelTraining(Step):
    """Model Training step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model training step."""
        # Set seed to replicate the results

        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Building and compiling a model...")

        # Put your model creation code here
        # You can log the model parameters and the final trained model in mlflow
        # You can also pass around the model uri using ioslots

        self.logger.info(f"The model has been trained and saved")
