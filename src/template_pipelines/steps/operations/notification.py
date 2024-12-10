"""Notifications step class."""

from aif.pyrogai.steps.step import Step  # noqa: E402


class NotificationStep(Step):
    """Preprocessing step."""

    def trigger_error(self):
        """This function will raise a ValueError when called."""
        raise ValueError("Intentional error triggered.")

    def run(self):
        """Run Notification step."""
        self.logger.info("Running Notification...")

        # Getting saved data from pipeline during mlflow run,
        # becasue of mlflow doesn't work we needed to add fake 'metrics'
        if self.platform.lower() != "vertex":
            data_from_root_run_id = self.mlflow.get_run(
                run_id=self.mlflow_utils.get_root_run_id()
            ).data
            metrics = data_from_root_run_id.metrics
        else:
            metrics = {}
            metrics["RMSE"] = 0.5

        # to trigger notification, change >= to <=
        if metrics["RMSE"] >= 1:
            self.trigger_error()

        self.logger.info("Notification is done.")
