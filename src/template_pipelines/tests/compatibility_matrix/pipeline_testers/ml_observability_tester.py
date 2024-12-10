"""Ml observability tester class."""
from logging import Logger

from compatibility_matrix_controller import CompatibilityMatrixController

from template_pipelines.tests.compatibility_matrix.pipeline_testers.base_tester import (
    IPipelineTester,
)


class MlObservabilityTester(IPipelineTester):
    """Ml observability tester class."""

    def __init__(self, logger: Logger, matrix_controller: CompatibilityMatrixController) -> None:
        """Init."""
        super().__init__(logger, matrix_controller)

    def prepare_env_to_run_pipeline(self) -> None:
        """Preparation env."""
        self.logger.info(
            f"Start preparing environment and files for {self.matrix_controller.pipeline_name}"
        )
        self.logger.info(
            f"Finish preparing environment and files for {self.matrix_controller.pipeline_name}"
        )
