"""Second integration test suite for dataset 20stocks."""

import logging
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from aif.pyrogai.ioutils.iocontext import IoContext
from template_pipelines.tests.steps.optimization.L3.l3_integration_test_suite import (
    L3IntegrationTestSuite,
)

logger = logging.getLogger(__name__)


class IntegrationTestSuite20StocksB(L3IntegrationTestSuite):
    """Second integration test suite for dataset 20stocks."""

    def run_tests(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ):
        """Run the test suite.

        Args:
            step_mlflow_run_id (str): run id of step
            pipeline_mlflow_run_id (str): run id of pipeline
            config (dict): step config, the same as in regular step
            aux_data_path (Optional[Path]): path to auxiliary data folder (input ioslot)
            ioctx (IoContext): pyrogai iocontext reference
        """
        logger.info("running L3 integration test suite 20stocksB")
        tests = [self._test_auxiliary_data_access, self._test_processing_time_limit]
        self._run_tests_with_late_failure(
            tests, step_mlflow_run_id, pipeline_mlflow_run_id, config, aux_data_path, ioctx
        )

    def _test_processing_time_limit(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ):
        """Test expecation that for large data processing time should be below 1 minute."""
        logger.info("running L3 integration test - test_processing_time_limit")

        pipeline_metrics = mlflow.MlflowClient().get_run(pipeline_mlflow_run_id).data.metrics

        if pipeline_metrics["solve time"] > config["expected_solve_time_upper_bound"]:
            logger.error("integration test suite failed")
            raise ValueError(
                f"Solve time {pipeline_metrics['solve time']} exceeds expected upper "
                f"bound {config['expected_solve_time_upper_bound']}"
            )

    def _test_auxiliary_data_access(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ):
        """Sample test if auxiliary dataset can be loaded and is accessible."""
        logger.info("running L3 integration test - test_auxiliary_data_access")

        logger.info(f"aux_data_path: {aux_data_path}")
        if not aux_data_path:
            raise ValueError(
                f"Auxiliary data access test requires a valid aux data path, received: {aux_data_path}"
            )

        # its possible to read any auxiliary data this way
        df = pd.read_csv(aux_data_path / "abc.csv", header=None)
        logger.info(df)
