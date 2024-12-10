"""L2 test for template_pipelines/steps/optimization/formulate_and_solve.py."""
import shutil
from pathlib import Path

import pytest

from dnalib.optimization_models.xpress_utils.testing import compare_solution
from template_pipelines.steps.optimization.formulate_and_solve import FormulateAndSolve
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet, load_tables
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


def test_20stocks(fixture_step):
    """Test the Formulate and Solve step.

    The test using 20 stocks inputs in portfolio_stocks.parquet in this folder and using
    runtime parameters above This will check the results stored in ioctx (fraction.parquet
    and buy.parquet).
    """
    # load reference solution
    reference_solution, _ = load_tables(
        [TEST_DATA_DIR / "buy.parquet", TEST_DATA_DIR / "fraction.parquet"]
    )

    # execute
    fixture_step.process()

    # load all parquet files in config['solution_tmp_dir']
    file_paths = fixture_step.ioctx.get_fns(f"{fixture_step.config['solution_tmp_dir']}/*.parquet")
    results, _ = load_tables(file_paths)

    # assert results
    compare_solution(
        test_solution=results,
        reference_solution=reference_solution,
        tolerance={"buy": 0.001, "fraction": 0.00001},
    )
