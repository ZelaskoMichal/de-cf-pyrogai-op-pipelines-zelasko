"""Unit tests for template_pipelines/steps/optimization/postprocess_data.py."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from template_pipelines.steps.optimization.postprocess_data import PostprocessData
from template_pipelines.tests.steps.optimization.constants import (
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import (
    copy_data_to_parquet,
    read_excel_or_csv_with_na,
)
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test

TESTED_MODULE_PATH = "template_pipelines.steps.optimization.postprocess_data.%s"


@pytest.fixture(scope="function")
def step():
    """Fixture returns PostprocessData step for unit tests.

    Step is initialized in isolation from dependencies as much as possible.
    """
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = PostprocessData()
        yield step


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns PostprocessData step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="postprocess_data",
    )

    step = PostprocessData()

    # prepare data for step in icotx
    # raw data
    copy_data_to_parquet(
        file_paths=[
            TEST_DATA_DIR / "input_data" / "stocks.csv",
            TEST_DATA_DIR / "input_data" / "regions.csv",
            TEST_DATA_DIR / "input_data" / "industries.csv",
        ],
        dest_path=Path(step.ioctx.get_output_fn(step.config["input_tmp_dir"])),
    )
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
        ],
        dest_path=Path(step.ioctx.get_output_fn(step.config["solution_tmp_dir"])),
    )

    yield step


@pytest.fixture(scope="function")
def dataframes_dict():
    """Fixture for data."""
    yield {
        "solution_df": pd.read_parquet(TEST_DATA_DIR / "output_data" / "fraction.parquet"),
        "solution_df_with_wrong_values": pd.read_parquet(
            TEST_DATA_DIR / "output_data" / "fraction_with_wrong_values.parquet"
        ),
        "portfolio_df": pd.read_csv(TEST_DATA_DIR / "input_data" / "portfolio_stocks.csv"),
        "raw_industry_df": pd.read_csv(TEST_DATA_DIR / "input_data" / "industries.csv"),
        "raw_industry_df_with_duplicates": pd.read_csv(
            TEST_DATA_DIR / "input_data" / "industries_with_duplicates.csv"
        ),
        "raw_industry_df_without_data": pd.read_csv(
            TEST_DATA_DIR / "input_data" / "industries_without_data.csv"
        ),
        "raw_region_df": pd.read_csv(TEST_DATA_DIR / "input_data" / "regions.csv"),
        "raw_region_df_with_duplicate": pd.read_csv(
            TEST_DATA_DIR / "input_data" / "regions_with_duplicates.csv"
        ),
        "raw_stocks_df": pd.read_csv(TEST_DATA_DIR / "input_data" / "stocks.csv"),
        "output_df": pd.read_csv(TEST_DATA_DIR / "output_data" / "output.csv", index_col=0),
    }


def test_process(step_semi_integrated):
    """Test that no errors occur, check output.parquet file."""
    # prepare
    output_dir_path = Path(
        step_semi_integrated.ioctx.get_output_fn(step_semi_integrated.config["output_tmp_dir"])
    )
    reference_output_df = read_excel_or_csv_with_na(
        TEST_DATA_DIR / "output_data" / "output.csv", index_col=[0]
    )

    # execute
    step_semi_integrated.process()

    # assert
    # result df
    result_output_df = pd.read_parquet(output_dir_path / "output.parquet")
    pd.testing.assert_frame_equal(result_output_df, reference_output_df)


@pytest.mark.parametrize(
    "solution_df, portfolio_df, raw_industry_df, raw_region_df, raw_stocks_df, expected_output_df",
    [
        (
            "solution_df",
            "portfolio_df",
            "raw_industry_df",
            "raw_region_df",
            "raw_stocks_df",
            "output_df",
        ),  # correct data
        (
            "solution_df",
            "portfolio_df",
            "raw_industry_df_with_duplicates",
            "raw_region_df_with_duplicate",
            "raw_stocks_df",
            "output_df",
        ),  # duplicates in industries and regions tables,
        (
            "solution_df_with_wrong_values",
            "portfolio_df",
            "raw_industry_df",
            "raw_region_df",
            "raw_stocks_df",
            "output_df",
        ),  # negative values in solutions to check filtering stocks by recommended portfolio percent greater than 0
    ],
)
@patch(TESTED_MODULE_PATH % "load_tables")
@patch(TESTED_MODULE_PATH % "StocksPortfolioSDM")
def test_run(
    mocked_sdm,
    mocked_load_tables,
    step,
    dataframes_dict,
    solution_df,
    portfolio_df,
    raw_industry_df,
    raw_region_df,
    raw_stocks_df,
    expected_output_df,
):
    """Test run method in multiple scenarios."""
    # We use load_sdm function twice in step's run method.
    mocked_load_tables.side_effect = [
        ({"fraction": dataframes_dict[solution_df]}, None),  # load solution
        (
            {
                "industries": dataframes_dict[raw_industry_df],
                "regions": dataframes_dict[raw_region_df],
                "stocks": dataframes_dict[raw_stocks_df],
            },
            None,
        ),  # load raw_data
    ]
    mocked_sdm.return_value.sdm_data = {"portfolio_stocks": dataframes_dict[portfolio_df]}
    step.save_output = Mock()
    step.run()
    df = step.save_output.call_args_list[0][0][0]
    assert_frame_equal(df, dataframes_dict[expected_output_df])
