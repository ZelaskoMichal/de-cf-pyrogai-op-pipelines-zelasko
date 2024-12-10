"""Unit tests for template_pipelines/steps/preprocessed_critical_custom_dq.py."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from template_pipelines.steps.optimization.preprocessed_critical_custom_dq import (
    PreprocessedCriticalCustomDq,
)
from template_pipelines.tests.steps.optimization.constants import (
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import (
    cast_numeric_runtime_parameters,
    copy_data_to_parquet,
)
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

TESTED_MODULE_PATH = "template_pipelines.steps.optimization.preprocessed_critical_custom_dq.%s"


@pytest.fixture(scope="function")
def step():
    """Fixture returns PreprocessedCriticalCustomDq step for unit tests.

    Step is initialized in isolation from dependencies as much as possible.
    """
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = PreprocessedCriticalCustomDq()
        step.runtime_parameters = cast_numeric_runtime_parameters(dict(RUNTIME_PARAMETERS))
        step.config = {
            "data_dir": "my_data_dir",
            "sdm_tmp_dir": "my_sdm_tmp_dir",
            "solution_tmp_dir": "my_solution_tmp_dir",
        }
        yield step


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns PreprocessedCriticalCustomDq step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="preprocessed_critical_custom_dq",
    )

    step = PreprocessedCriticalCustomDq()

    dest_ioctx_path = Path(step.ioctx.get_output_fn("sdm"))

    copy_data_to_parquet(
        file_paths=[TEST_DATA_DIR / "input_data" / "portfolio_stocks.parquet"],
        dest_path=dest_ioctx_path,
    )

    yield step


def test_process(step_semi_integrated):
    """Test that no errors occur."""
    # execute
    step_semi_integrated.process()


@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_fail_due_to_too_few_rows_for_max_fraction_condition(mocked_sdm, step):
    """Test run()."""
    # prepare
    portfolio_stocks_table = pd.DataFrame(
        {
            "is_risky": [0, 1, 1],
            "region": ["r1", "r1", "r1"],
            "name": ["n1", "n2", "n3"],
        }
    )
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}

    # execute & assert
    with pytest.raises(ValueError, match="Too few rows to meet max fraction constraints"):
        step.run()


@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_fail_due_to_too_few_rows_in_each_region(mocked_sdm, step):
    """Test run()."""
    # prepare
    portfolio_stocks_table = pd.DataFrame(
        {
            "is_risky": [0, 1, 1, 1, 1],
            "region": ["r1", "r2", "r2", "r2", "r2"],
            "name": ["n1", "n2", "n3", "n4", "n5"],
        }
    )
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}

    # execute & assert
    with pytest.raises(ValueError, match="Too few rows for region: r1"):
        step.run()


@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_fail_due_to_not_met_two_conditions_at_once(mocked_sdm, step):
    """Test run()."""
    # prepare
    portfolio_stocks_table = pd.DataFrame(
        {
            "is_risky": [0, 0],
            "region": ["r1", "r2"],
            "name": ["n1", "n2"],
        }
    )
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}

    # execute & assert
    with pytest.raises(
        ValueError,
        match="Too few rows to meet max fraction constraints; Too few rows for region: r1, r2",
    ):
        step.run()
