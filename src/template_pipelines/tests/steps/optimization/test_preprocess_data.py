"""Unit tests for preprocessing step."""
from pathlib import Path

import pandas as pd
import pytest

from template_pipelines.steps.optimization.preprocess_data import PreprocessData
from template_pipelines.tests.steps.optimization.constants import (
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns PreprocessData step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="preprocess_data",
    )

    step = PreprocessData()

    dest_ioctx_path = Path(step.ioctx.get_output_fn("input"))

    copy_data_to_parquet(
        file_paths=[
            TEST_DATA_DIR / "input_data" / "stocks.csv",
            TEST_DATA_DIR / "input_data" / "regions.csv",
            TEST_DATA_DIR / "input_data" / "industries.csv",
            TEST_DATA_DIR / "input_data" / "general_inputs.parquet",
        ],
        dest_path=dest_ioctx_path,
    )

    yield step


def test_process(step_semi_integrated):
    """Test that no errors occur, resulted portfolio_stocks df and mlflow metrics."""
    # prepare
    reference_portfolio_df = pd.read_parquet(
        TEST_DATA_DIR / "input_data" / "portfolio_stocks.parquet"
    )

    # execute
    step_semi_integrated.process()

    # assert
    # result df
    result_portfolio_df = pd.read_parquet(
        Path(step_semi_integrated.ioctx.get_output_fn("sdm")) / "portfolio_stocks.parquet"
    )
    pd.testing.assert_frame_equal(result_portfolio_df, reference_portfolio_df)
    # mlflow metrics
    metrics = (
        step_semi_integrated.mlflow.MlflowClient()
        .get_run(step_semi_integrated.active_mflow_run_id)
        .data.metrics
    )
    assert metrics == {
        "general_inputs_rows": 2,
        "industries_rows": 7,
        "stocks_rows": 20,
        "regions_rows": 3,
    }
