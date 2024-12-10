"""Unit tests for template_pipelines/steps/optimization/solution_custom_dq.py."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pandas as pd
import pytest

from template_pipelines.steps.optimization.solution_custom_dq import SolutionCustomDq
from template_pipelines.tests.steps.optimization.constants import (
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

TESTED_MODULE_PATH = "template_pipelines.steps.optimization.solution_custom_dq.%s"


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns SolutionCustomDq step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="solution_custom_dq",
    )

    step = SolutionCustomDq()

    # prepare data for step in icotx
    # sdm data
    copy_data_to_parquet(
        file_paths=[
            TEST_DATA_DIR / "input_data" / "portfolio_stocks.parquet",
        ],
        dest_path=Path(step.ioctx.get_output_fn(step.config["sdm_tmp_dir"])),
    )
    # solution data
    copy_data_to_parquet(
        file_paths=[
            TEST_DATA_DIR / "output_data" / "fraction.parquet",
            TEST_DATA_DIR / "output_data" / "buy.parquet",
        ],
        dest_path=Path(step.ioctx.get_output_fn(step.config["solution_tmp_dir"])),
    )

    yield step


@pytest.fixture(scope="module")
def step_and_output_file():
    """Fixture returns SolutionCustomDq step for unit tests.

    Step is initialized in isolation from dependencies as much as possible. Output file is a
    temporary file substituted for the outputs file
    """
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        output_file = NamedTemporaryFile(delete=False)
        output_file.close()

        # bind the temporary file from step with file from test
        with patch(TESTED_MODULE_PATH % "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.__enter__.return_value.name = output_file.name
            step = SolutionCustomDq()
            step.outputs = {}
            step.runtime_parameters = RUNTIME_PARAMETERS
            step.config = {
                "data_dir": "my_data_dir",
                "sdm_tmp_dir": "my_sdm_tmp_dir",
                "solution_tmp_dir": "my_solution_tmp_dir",
                "dq_warnings_logger_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
            yield step, output_file

        # remove the temporary file
        os.remove(output_file.name)


def test_process(step_semi_integrated):
    """Test that no errors occur."""
    # execute
    step_semi_integrated.process()

    # assert
    assert "solution_dq_warnings.log" in step_semi_integrated.outputs


@patch(TESTED_MODULE_PATH % "load_tables")
@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_in_case_of_critical_error(mocked_sdm, mocked_load_tables, step_and_output_file):
    """Test run method in case of critical error."""
    # prepare
    portfolio_stocks_table = pd.DataFrame({"a": [0, 1, 2]})
    buy_table = pd.DataFrame({"b": [2, 3]})
    fraction_table = pd.DataFrame({"c": [4, 5]})
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}
    mocked_load_tables.side_effect = [({"buy": buy_table},), ({"fraction": fraction_table},)]
    step, _ = step_and_output_file

    # execute && assert
    with pytest.raises(
        ValueError, match=f"buy solution doesn't have the same number of rows as portfolio_stocks"
    ):
        step.run()


@patch(TESTED_MODULE_PATH % "load_tables")
@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run_in_case_of_warning(mocked_sdm, mocked_load_tables, step_and_output_file):
    """Test run method in case of warning."""
    # prepare
    portfolio_stocks_table = pd.DataFrame(
        {"name": ["foo", "bar", "baz"]}
    )  # Some stocks will make more than 15% of your portfolio
    buy_table = pd.DataFrame({"buy": [1, 1, 1], "name": ["foo", "bar", "baz"]})
    fraction_table = pd.DataFrame({"fraction": [0.16, 0.16, 0.16], "name": ["foo", "bar", "baz"]})
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": portfolio_stocks_table}
    mocked_load_tables.return_value = ({"fraction": fraction_table, "buy": buy_table},)
    step, output_file = step_and_output_file

    # execute
    step.run()

    #  assert
    with open(output_file.name, "r") as file:
        log_contents = file.read()
        assert (
            "Some stocks will make more than 15% of your portfolio: ['foo', 'bar', 'baz']"
            in log_contents
        )
