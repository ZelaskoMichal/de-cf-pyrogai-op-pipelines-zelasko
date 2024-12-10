"""Unit tests for template_pipelines/steps/preprocessed_warning_custom_dq.py."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pandas as pd
import pytest

from template_pipelines.steps.optimization.preprocessed_warning_custom_dq import (
    PreprocessedWarningCustomDq,
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

TESTED_MODULE_PATH = "template_pipelines.steps.optimization.preprocessed_warning_custom_dq.%s"


@pytest.fixture(scope="function")
def step_and_output_file():
    """Fixture returns PreprocessedWarningCustomDq step for unit tests and output file.

    Step is initialized in isolation from dependencies as much as possible. Output file is a
    temporary file substituted for the outputs file
    """
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        # create the temporary file to check if some logs are saved to file
        output_file = NamedTemporaryFile(delete=False)
        output_file.close()

        # bind the temporary file from step with file from test
        with patch(TESTED_MODULE_PATH % "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.__enter__.return_value.name = output_file.name

            # prepare step
            step = PreprocessedWarningCustomDq()
            step.outputs = {}
            step.runtime_parameters = cast_numeric_runtime_parameters(dict(RUNTIME_PARAMETERS))
            step.config = {
                "data_dir": "my_data_dir",
                "sdm_tmp_dir": "my_sdm_tmp_dir",
                "solution_tmp_dir": "my_solution_tmp_dir",
                "dq_warnings_logger_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
            yield step, output_file

        # remove the temporary file
        os.remove(output_file.name)


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns PreprocessedWarningCustomDq step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="preprocessed_warning_custom_dq",
    )

    step = PreprocessedWarningCustomDq()

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
def test_run_warn_due_to_there_is_no_risky_stock(mocked_sdm, step_and_output_file, caplog):
    """Test step's run method in case of risky stocks warning."""
    # prepare
    caplog.clear()

    step, output_file = step_and_output_file

    portfolio_stocks_table = pd.DataFrame(
        {
            "is_risky": [0, 0, 0],
            "region": ["r1", "r1", "r2"],
            "name": ["n1", "n2", "n3"],
        }
    )
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}

    # execute
    step.run()

    # assert
    assert "There is no risky stock." in [r.msg for r in caplog.records]
    assert "dq_warnings.log" in step.outputs
    with open(output_file.name, "r") as file:
        log_contents = file.read()
        assert "There is no risky stock." in log_contents


@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_warn_due_to_only_one_region(mocked_sdm, step_and_output_file, caplog):
    """Test step's run method in case of 1 region warning."""
    # prepare
    caplog.clear()
    step, output_file = step_and_output_file
    portfolio_stocks_table = pd.DataFrame(
        {
            "is_risky": [1, 1],
            "region": ["r1", "r1"],
            "name": ["n1", "n2"],
        }
    )
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}

    # execute
    step.run()

    # assert
    assert "There is only 1 region." in [r.msg for r in caplog.records]
    assert "dq_warnings.log" in step.outputs
    with open(output_file.name, "r") as file:
        log_contents = file.read()
        assert "There is only 1 region." in log_contents
