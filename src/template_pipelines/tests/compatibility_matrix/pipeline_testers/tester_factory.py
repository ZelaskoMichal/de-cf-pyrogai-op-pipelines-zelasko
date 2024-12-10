"""Factory testers."""
import logging

from compatibility_matrix_controller import CompatibilityMatrixController

from template_pipelines.tests.compatibility_matrix.pipeline_testers.base_tester import (
    IPipelineTester,
)
from template_pipelines.tests.compatibility_matrix.pipeline_testers.ml_iris_tester import (
    MlIrisTester,
)
from template_pipelines.tests.compatibility_matrix.pipeline_testers.ml_observability_tester import (
    MlObservabilityTester,
)
from template_pipelines.tests.compatibility_matrix.pipeline_testers.ml_skeleton_tester import (
    MlSkeletonTester,
)
from template_pipelines.tests.compatibility_matrix.pipeline_testers.ml_training_tester import (
    MlTrainingTester,
)
from template_pipelines.tests.compatibility_matrix.pipeline_testers.upload_data_tester import (
    UploadDataTester,
)


class PipelineTesterFactory:
    """PipelineTesterFactory."""

    @staticmethod
    def get_preparator(
        logger: logging.Logger, matrix_controller: CompatibilityMatrixController
    ) -> IPipelineTester:
        """Get pipeline tester based on pipeline name."""
        pipeline_name = matrix_controller.pipeline_name

        if pipeline_name == "ml_iris":
            return MlIrisTester(logger, matrix_controller)
        elif pipeline_name == "ml_skeleton":
            return MlSkeletonTester(logger, matrix_controller)
        elif pipeline_name == "ml_observability":
            return MlObservabilityTester(logger, matrix_controller)
        elif pipeline_name == "upload_data":
            return UploadDataTester(logger, matrix_controller)
        elif pipeline_name == "ml_training":
            return MlTrainingTester(logger, matrix_controller)

        msg = f"No preparator found for {pipeline_name}"
        logger.error(msg)
        raise RuntimeError(msg)
