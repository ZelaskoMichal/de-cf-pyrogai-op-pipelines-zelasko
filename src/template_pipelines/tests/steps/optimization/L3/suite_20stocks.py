"""Integration test suite for 20stocks dataset."""

import logging
from pathlib import Path
from typing import Optional

import mlflow

from aif.pyrogai.ioutils.iocontext import IoContext
from template_pipelines.tests.steps.optimization.L3.l3_integration_test_suite import (
    L3IntegrationTestSuite,
)

logger = logging.getLogger(__name__)


class IntegrationTestSuite20Stocks(L3IntegrationTestSuite):
    """Integration test suite for 20stocks dataset."""

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
        logger.info("running L3 integration test suite 20stocks")
        tests = [self._test_formulation_build_time, self._test_objective_function_tolerance]
        self._run_tests_with_late_failure(
            tests, step_mlflow_run_id, pipeline_mlflow_run_id, config, aux_data_path, ioctx
        )

    def _test_objective_function_tolerance(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ):
        """Test if objective function value is within range."""
        logger.info("running L3 integration test - test_objective_function_tolerance")
        pipeline_metrics = mlflow.MlflowClient().get_run(pipeline_mlflow_run_id).data.metrics

        if (
            abs(pipeline_metrics["optimal objective Value"] - config["expected_objfn_value"])
            > config["objfn_value_tol"]
        ):
            logger.error(
                f"Difference of objective function value {pipeline_metrics['optimal objective Value']} "
                f"and expected value {config['expected_objfn_value']} "
                f"exceeds allowed tolerance {config['objfn_value_tol']}"
            )

    def _test_formulation_build_time(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ):
        """Test if formulation build time is below limit."""
        logger.info("running L3 integration test - test_formulation_build_time")
        pipeline_metrics = mlflow.MlflowClient().get_run(pipeline_mlflow_run_id).data.metrics

        build_time = pipeline_metrics["build time"]
        max_build_time = config["max_formulation_build_time"]

        if build_time > max_build_time:
            raise ValueError(f"Build time {build_time} exceeds allowed limit {max_build_time}")
