"""Main file with workflow to generate matrix."""

import logging
import os
from pathlib import Path
from typing import Generator, Optional

from template_pipelines.tests.compatibility_matrix.compatibility_matrix_controller import (
    CompatibilityMatrixController,
)
from template_pipelines.tests.compatibility_matrix.logger import get_configured_logger
from template_pipelines.tests.compatibility_matrix.pipeline_testers.tester_factory import (
    PipelineTesterFactory,
)


def get_environment_variables() -> tuple:
    """Retrieve and validate required environment variables."""
    pyrogai_version: Optional[str] = os.getenv("PYROGAI_VERSION")
    pipeline_name: Optional[str] = os.getenv("PIPELINE_NAME")
    env_tags: str = os.getenv("TPT_TAGS", "")
    tpt_tags: Generator[str, None, None] = (
        tag.replace("]", "") for tag in env_tags.replace('"', "").split(",") if tag.startswith("v")
    )

    missing_vars = [
        var_name
        for var_name, var_value in {
            "PYROGAI_VERSION": pyrogai_version,
            "PIPELINE_NAME": pipeline_name,
            "TPT_TAGS": tpt_tags,
        }.items()
        if var_value is None
    ]

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return pyrogai_version, pipeline_name, tpt_tags


def main() -> None:
    """Main function to test compatibility matrix, here everything starts."""
    logger = get_configured_logger("COMPATIBILITY_MATRIX_LOGGER", "my_logs.log", logging.DEBUG)
    logger.info("START TESTING PIPELINES ON TEMPLATE PIPELINES")

    try:
        pyrogai_version, pipeline_name, tpt_tags = get_environment_variables()
    except EnvironmentError as e:
        logger.error(e)
        return

    matrix_controller = CompatibilityMatrixController(
        compatibility_matrix_file_path=Path("compatibility_matrix.json"),
        logger=logger,
        tpt_tags=tpt_tags,
        pyrogai_version=pyrogai_version,
        pipeline_name=pipeline_name,
    )

    pipeline_tester = PipelineTesterFactory.get_preparator(logger, matrix_controller)
    pipeline_tester.prepare_and_test()

    logger.info("FINISH TESTING PIPELINES ON TEMPLATE PIPELINES")


if __name__ == "__main__":
    main()
