"""L0 tests for template_pipelines/steps/optimization/formulate_and_solve.py."""
import shutil
from pathlib import Path

import pytest

from template_pipelines.steps.optimization.formulate_and_solve import FormulateAndSolve
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

CONFIG_FILE = "config.json"
TEST_CONFIG_MODULE = "template_pipelines.tests.steps.optimization.config"
TEST_DATA_DIR = Path(__file__).parent


@pytest.fixture(scope="function")
def fixture_step():
    """Fixture for FormulateAndSolve step."""
    # prepare provider
    runtime_parameters = {
        "max_risky_stocks": 3,
        "max_risky_stocks_ratio": 0.25,
        "max_ratio_per_stock": 0.3,
        "min_ratio_per_stock": 0.01,
        "min_ratio_per_region": 0.2,
        "min_stocks_per_region": 2,
        "max_total_stocks": 8,
        "max_number_risky_sum_activation": "hard",
        "min_number_per_region_activation": "soft",
    }

    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="test_formulate_and_solve_step",
        runtime_parameters=runtime_parameters,
        step_name="formulate_and_solve",
    )

    step = FormulateAndSolve()

    # load input date to ioctx
    dest_ioctx_path = Path(step.ioctx.get_output_fn("sdm"))

    copy_data_to_parquet(
        file_paths=[TEST_DATA_DIR / "portfolio_stocks.parquet"],
        dest_path=dest_ioctx_path,
    )
    shutil.copy(TEST_DATA_DIR / "values.json", dest_ioctx_path)

    yield step


def test_smoke_test(fixture_step):
    """The Smoke test for the formulate and solve step.

    The test using 20 stocks inputs in portfolio_stocks.parquet in this folder and using
    runtime parameters above. This will check the if results are stored in ioctx
    (fraction.parquet and buy.parquet).
    """
    # execute
    fixture_step.process()

    # check if solution exists
    solution_dir_path = fixture_step.ioctx.get_output_fn(fixture_step.config["solution_tmp_dir"])
    assert (solution_dir_path / "buy.parquet").exists()
    assert (solution_dir_path / "fraction.parquet").exists()
