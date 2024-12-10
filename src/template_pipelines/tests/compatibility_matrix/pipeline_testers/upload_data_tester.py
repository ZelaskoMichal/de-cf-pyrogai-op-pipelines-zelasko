"""Upload data tester class."""

import subprocess
from logging import Logger

from compatibility_matrix_controller import CompatibilityMatrixController

from template_pipelines.tests.compatibility_matrix.pipeline_testers.base_tester import (
    IPipelineTester,
)


class UploadDataTester(IPipelineTester):
    """Upload data tester class."""

    def __init__(self, logger: Logger, matrix_controller: CompatibilityMatrixController) -> None:
        """Init."""
        super().__init__(logger, matrix_controller)

        self._already_created_files = False
        self._data_to_create_files = {
            "campaigns": {"campaign_id": [1, 2, 3, 4, 5]},
            "coupon_item_mapping": {"coupon_id": [1, 2, 3], "item_id": [1, 2, 3]},
            "coupon_redemption": {
                "id": [1, 2, 3],
                "redemption_status": [1, 0, 0],
                "campaign_id": [1, 2, 1],
                "coupon_id": [3, 2, 1],
                "customer_id": [1, 2, 3],
            },
            "customer_demographics": {"customer_id": [1, 2, 3]},
            "customer_transactions": {
                "customer_id": [1, 2, 3],
                "item_id": [3, 2, 1],
            },
            "items": {"item_id": [1, 2, 3]},
        }

    def prepare_env_to_run_pipeline(self):
        """Preparation env."""
        self.logger.info(
            f"Start preparing environment and files for {self.matrix_controller.pipeline_name}"
        )

        self.create_files_in_folder(self._data_to_create_files, "data")

        self.logger.info(
            f"Finish preparing environment and files for {self.matrix_controller.pipeline_name}"
        )

    def run_pipeline(self):
        """Func to run pipeline."""
        self.logger.info(f"Running pipeline {self.matrix_controller.pipeline_name}")
        try:
            subprocess.check_call(
                [
                    "aif",
                    "pipeline",
                    "run",
                    "--pipelines",
                    self.matrix_controller.pipeline_name,
                    "-p",
                    "dest_platform=Local",
                    "--config-module",
                    "template_pipelines.config",
                    "--environment",
                    "dev",
                ]
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred during TPT pipeline run: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg)
