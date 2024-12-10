"""L1 test of max_risky_stocks_ratio constraint."""
from pathlib import Path

import pandas as pd
import pytest

from dnalib.optimization_models.xpress_utils.testing import compare_solution
from template_pipelines.steps.optimization.formulate_and_solve import FormulateAndSolve
from template_pipelines.utils.optimization.io_utils import load_tables, save_tables, save_values
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

CONFIG_FILE = "config.json"
TEST_CONFIG_MODULE = "template_pipelines.tests.steps.optimization.config"
TEST_DATA_DIR = Path(__file__).parent


@pytest.fixture(scope="function")
def fixture_step():
    """Fixture for FormulateAndSolve step."""
    # prepare provider
    runtime_parameters = {
        "max_risky_stocks": 999999,
        "max_risky_stocks_ratio": 0.8,
        "max_ratio_per_stock": 0.65,
        "min_ratio_per_stock": 0.01,
        "min_ratio_per_region": 0,
        "min_stocks_per_region": 0,
        "max_total_stocks": 999999,
        "min_number_per_region_activation": "off",
        "max_number_risky_sum_activation": "off",
    }

    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="test_formulate_and_solve_step",
        runtime_parameters=runtime_parameters,
        step_name="formulate_and_solve",
    )

    step = FormulateAndSolve()
    yield step


def test_max_risky_stocks_ratio(fixture_step):
    """Tests the formulate and solve step with a reduced input data and custom runtime_parameters.

    This will check the results stored in ioctx (fraction.parquet and buy.parquet) and also modifies the path used by
    mlflow, to make the retrieval of results easy (solution.csv).

    This check verifies that max_risky_stocks_ratio is being applied.
    If it was not applied, the optimum would be to invest in only treasury and hardware.
    """
    # SDM
    df = pd.DataFrame(
        data={
            "name": ["treasury", "hardware", "grocery"],
            "expected_return": [12.0, 8.5, 2.3],
            "is_risky": [1, 1, 0],
            "region": ["EU", "EU", "EU"],
        }
    )
    dest_ioctx_path = Path(fixture_step.ioctx.get_output_fn("sdm"))
    save_tables({"portfolio_stocks": df}, dest_ioctx_path)
    other_data = {"penalty_max_number_risky": 0, "penalty_min_number_per_region": 0}
    save_values(other_data, dest_ioctx_path)

    # reference solution
    buy = pd.DataFrame(
        data={
            "name": ["treasury", "hardware", "grocery"],
            "buy": [1, 1, 1],
        }
    )
    fraction = pd.DataFrame(
        data={
            "name": ["treasury", "hardware", "grocery"],
            "fraction": [0.65, 0.15, 0.2],
        }
    )
    reference_solution = {"buy": buy, "fraction": fraction}

    # execute
    fixture_step.process()

    # load all parquet files in config['solution_tmp_dir'] and load solution
    file_paths = fixture_step.ioctx.get_fns(f"{fixture_step.config['solution_tmp_dir']}/*.parquet")
    results, _ = load_tables(file_paths)

    # assert
    compare_solution(
        test_solution=results,
        reference_solution=reference_solution,
        tolerance=0.0001,
    )