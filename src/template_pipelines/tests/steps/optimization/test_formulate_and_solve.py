"""Tests for template_pipelines/steps/optimization/formulate_and_solve.py."""
import shutil
from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import mlflow
import pandas as pd
import pytest
import xpress as xp

from aif.pyrogai.ioutils.iocontext import IoContext
from aif.pyrogai.pipelines.components.model import OptimizationConfig
from aif.pyrogai.steps.mlflow import MlflowUtils
from template_pipelines.steps.optimization.formulate_and_solve import FormulateAndSolve
from template_pipelines.tests.steps.optimization.constants import (  # noqa I202
    CONFIG,
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import (
    cast_numeric_runtime_parameters,
    copy_data_to_parquet,
    load_tables,
)
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

TESTED_MODULE_PATH = "template_pipelines.steps.optimization.%s"


@pytest.fixture(scope="function")
def step():
    """Fixture returns FormulateAndSolve step for unit tests.

    Step is initialized in isolation from dependencies as much as possible.
    """
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = FormulateAndSolve()
        step.runtime_parameters = cast_numeric_runtime_parameters(dict(RUNTIME_PARAMETERS))
        step.logger = Mock(Logger, autospec=True)
        step.mlflow = (
            Mock()
        )  # don't use autospec because installing uneeded packages will be required e.g. tensorflow
        step.mlflow.log_metrics = Mock(mlflow.log_metrics, autospec=True)
        step.mlflow_utils = Mock(MlflowUtils, autospec=True)
        step.ioctx = Mock(IoContext, autospec=True)
        step.ioctx.get_output_fn.return_value = Path("output_dir")

        step.config = MagicMock()
        step.config.__getitem__.side_effect = CONFIG.__getitem__
        step.config.keys.return_value = CONFIG.keys()
        step.config.model.optimization = {
            "formulate_and_solve": OptimizationConfig(
                fall_back_to_community_license=True, continue_pipeline_on_bad_solution=False
            )
        }

        step.xp = Mock()

    yield step


@pytest.fixture(scope="function")
def step_for_semi_integrated():
    """Fixture returns FormulateAndSolve step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="formulate_and_solve",
    )

    step = FormulateAndSolve()

    dest_ioctx_path = Path(step.ioctx.get_output_fn("sdm"))

    copy_data_to_parquet(
        file_paths=[TEST_DATA_DIR / "input_data" / "portfolio_stocks.parquet"],
        dest_path=dest_ioctx_path,
    )
    shutil.copy(src=TEST_DATA_DIR / "input_data" / "values.json", dst=dest_ioctx_path)

    yield step


def test_merge_and_log_slacks(step):
    """Test step's merge_and_log_slacks method in success case.

    Test whether  given dataframes in `slack_dict_of_dfs` will be merged and logged as one
    dataframe as expected.
    """
    # prepare
    slack_dict_of_dfs = {
        "invest_all": pd.DataFrame(data={"invest_all": [5.551115123125783e-17]}),
        "tie_buy_and_fraction_lb": pd.DataFrame(
            data={"name": ["treasury", "cars"], "tie_buy_and_fraction_lb": [1.0, 2.11]}
        ),
        "tie_buy_and_fraction_ub": pd.DataFrame(
            data={"name": ["treasury", "cars"], "tie_buy_and_fraction_ub": [33.1, -1.0]}
        ),
        "max_number_risky_sum": pd.DataFrame(data={"max_number_risky_sum": [1.0]}),
        "max_fraction_risky": pd.DataFrame(data={"max_fraction_risky": [4.6]}),
        "min_fraction_per_region": pd.DataFrame(
            data={"region": ["EU"], "min_fraction_per_region": [0.3]}
        ),
        "min_number_per_region": pd.DataFrame(
            data={"region": ["EU"], "min_number_per_region": [8.0]}
        ),
        "max_number_stocks": pd.DataFrame(data={"max_number_stocks": [2.0]}),
    }

    # execute
    step.merge_and_log_slacks(slack_dict_of_dfs)

    # assert
    _, kwargs = step.mlflow_utils.handle_logging_artifact.call_args
    output_df = kwargs["df"]
    output_df.fillna(value="NaN", inplace=True)

    assert kwargs["file_name"] == "slacks.csv"
    assert kwargs["mlflow_artifact_path"] == "solved"
    assert output_df.to_dict("list") == {
        "stock": ["treasury", "cars", "NaN", "NaN", "NaN", "NaN", "NaN"],
        "tie_buy_and_fraction_lb": [1.0, 2.11, "NaN", "NaN", "NaN", "NaN", "NaN"],
        "tie_buy_and_fraction_ub": [33.1, -1.0, "NaN", "NaN", "NaN", "NaN", "NaN"],
        "region": ["NaN", "NaN", "EU", "NaN", "NaN", "NaN", "NaN"],
        "min_number_per_region": ["NaN", "NaN", 8.0, "NaN", "NaN", "NaN", "NaN"],
        "min_fraction_per_region": ["NaN", "NaN", 0.3, "NaN", "NaN", "NaN", "NaN"],
        "constraint_name": [
            "NaN",
            "NaN",
            "NaN",
            "invest_all",
            "max_number_risky_sum",
            "max_fraction_risky",
            "max_number_stocks",
        ],
        "slack": ["NaN", "NaN", "NaN", 5.551115123125783e-17, 1.0, 4.6, 2.0],
    }


@patch(TESTED_MODULE_PATH % "formulate_and_solve.os.makedirs")
@patch("pandas.DataFrame.to_parquet")
def test_save_solution(mocked_pd_to_to_parquet, mocked_os_makedirs, step):
    """Test step's save_solution method.

    Check if each dataframe in given dict of dataframes is saved separately to expected
    location as file, and if correct mlflow artifact is created
    """
    # prepare
    output_data = {
        "fraction": pd.DataFrame({"name": ["foo"], "a": [1]}),
        "buy": pd.DataFrame({"name": ["foo"], "b": [2]}),
    }
    step.ioctx.get_output_fn.return_value = Path("mocked_path")
    step.outputs = {}
    step.mlflow_utils.handle_logging_artifact.return_value = "output_mlflow_uri"

    # execute
    step.save_solution(output_data)

    # assert
    # saving merged dataframes as mlflow artifact
    mocked_method = step.mlflow_utils.handle_logging_artifact
    mocked_method.assert_called_once()
    assert mocked_method.call_args.kwargs["file_name"] == "solution.csv"
    assert mocked_method.call_args.kwargs["mlflow_artifact_path"] == "solved"
    pd.testing.assert_frame_equal(
        mocked_method.call_args.kwargs["df"], pd.DataFrame({"name": ["foo"], "a": [1], "b": [2]})
    )
    # saving each dataframe as file
    assert mocked_pd_to_to_parquet.call_count == 2
    assert (
        mocked_pd_to_to_parquet.call_args_list[0][0][0] == Path("mocked_path") / "fraction.parquet"
    )
    assert mocked_pd_to_to_parquet.call_args_list[1][0][0] == Path("mocked_path") / "buy.parquet"


@patch(TESTED_MODULE_PATH % "formulate_and_solve.IncludeXpressLogs")
def test_init_xpress_problem(mocked_xpress_logs, step):
    """Test step's init_xpress_proble method."""
    # prepare
    step.xp = xp

    # execute
    xpress_problem = step.init_xpress_problem()

    # assert
    mocked_xpress_logs.assert_called_once()
    assert step.xpress_problems_dict == {"portfolio": xpress_problem}


def test_set_solver_controls(step):
    """Test step's set_solver_controls method."""
    # prepare
    step.xp = xp
    xpress_problem = xp.problem()

    # execute
    step.set_solver_controls(xpress_problem)

    # assert
    step.mlflow_utils.log.assert_called_once()
    assert xpress_problem.getControl("miprelstop") == 1e-4


@patch(TESTED_MODULE_PATH % "formulate_and_solve.StocksPortfolioSDM")
def test_load_sdm(mocked_sdm, step):
    """Test step's load_sdm method."""
    # mock
    sdm_data = {"my_table": pd.DataFrame({"a": [1]}), "my_value": 1.1}
    mocked_sdm.return_value.sdm_data = sdm_data
    mocked_sdm.return_value.items.return_value = sdm_data

    # execute
    tables, tables_metrics, values, values_metrics = step.load_sdm()

    # assert
    assert tables["my_table"].equals(sdm_data["my_table"])
    assert tables_metrics == {"my_table_rows": 1}
    assert values["my_value"] == sdm_data["my_value"]
    assert values_metrics == {"num_loaded_values": 1}


def test_get_table_statistics(step):
    """Test step's get_table_statistics method."""
    # prepare
    data = pd.DataFrame({"a": [0]})

    # execute
    metrics = step.get_table_statistics("my_table", data)

    # assert
    assert metrics == {"my_table_rows": 1}


def test_process(step_for_semi_integrated):
    """Test step's flow by calling process method.

    Test goes through formulating and solving optimization using Xpress. It is checked here:
    metrics, output data from solved problem and saving data to files.
    """
    # prepare
    expected_non_time_metrics_with_values = {
        "MIP entities": 20.0,
        "columns": 43.0,
        "is mip": 1.0,
        "matrix nonzero entries": 185.0,
        "mip gap": 0.0,
        "optimal objective Value": 11.729,
        "original MIP entities": 20.0,
        "original columns": 43.0,
        "original rows": 50.0,
        "portfolio_stocks_rows": 20.0,
        "relative optimality gap": 0.0,
        "rows": 50.0,
        "solution status": 1.0,
        "solve status": 3.0,
        "stop status": 0.0,
        "num_loaded_values": 2,
    }  # values for keys related to time will be always different
    # so it makes no sense to check values

    expected_params_with_values = {
        "max_number_risky_sum_activation": "hard",
        "max_ratio_per_stock": "0.3",
        "max_risky_stocks": "3",
        "max_risky_stocks_ratio": "0.25",
        "max_total_stocks": "8",
        "min_number_per_region_activation": "soft",
        "min_ratio_per_region": "0.2",
        "min_ratio_per_stock": "0.01",
        "min_stocks_per_region": "2",
        "miprelstop": "0.0001",
    }
    expected_solution_artifact = pd.read_csv(TEST_DATA_DIR / "output_data" / "solution.csv")
    expected_slacks_artifact = pd.read_csv(TEST_DATA_DIR / "output_data" / "slacks.csv")

    expected_fraction_solution = pd.read_parquet(TEST_DATA_DIR / "output_data" / "fraction.parquet")
    expected_buy_solution = pd.read_parquet(TEST_DATA_DIR / "output_data" / "buy.parquet")

    solution_file_paths = step_for_semi_integrated.ioctx.get_fns(
        f"{step_for_semi_integrated.config['solution_tmp_dir']}/*.parquet"
    )

    # execute
    step_for_semi_integrated.process()

    # assert, that ioctx and mlflow outputs are correct
    run_id = step_for_semi_integrated.active_mflow_run_id
    # ioctx - solution
    solution_dict_of_dfs, _ = load_tables(solution_file_paths)
    pd.testing.assert_frame_equal(solution_dict_of_dfs["fraction"], expected_fraction_solution)
    pd.testing.assert_frame_equal(solution_dict_of_dfs["buy"], expected_buy_solution)
    # mlflow - artifacts
    result_solution_artifact = pd.read_csv(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="solved/solution.csv")
    )
    result_slacks_artifact = pd.read_csv(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="solved/slacks.csv")
    )
    pd.testing.assert_frame_equal(result_solution_artifact, expected_solution_artifact)
    pd.testing.assert_frame_equal(result_slacks_artifact, expected_slacks_artifact)
    # mlflow - metrics && params
    pipeline_run_data = step_for_semi_integrated.mlflow.MlflowClient().get_parent_run(run_id).data
    result_metrics = pipeline_run_data.metrics
    # metrics
    assert result_metrics.pop("build time")
    assert result_metrics.pop("solve time")
    assert result_metrics == expected_non_time_metrics_with_values
    # params
    assert pipeline_run_data.params == expected_params_with_values
