"""SDM tests."""

from unittest.mock import Mock, call, patch

import pandas as pd

from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.tests.steps.optimization.preprocessing.utils import get_sdm_io_tables


@patch("template_pipelines.steps.optimization.preprocessing.sdm.load_tables")
@patch("template_pipelines.steps.optimization.preprocessing.sdm.load_values")
def test_load_stored_sdm(load_values_mock, load_tables_mock):
    """Test that SDM loading logic succeeds."""
    # prepare
    ioctx_mock = Mock()
    file_path = "ioctx/path/file.parquet"
    values_path = "ioctx/path/values.json"
    ioctx_mock.get_fns.side_effect = [file_path, values_path]
    df = pd.DataFrame(data={"foo": [1, 2]})
    load_tables_mock.return_value = ({"file": df}, {"file": file_path})
    load_values_mock.return_value = {"param1": 1}

    sdm = StocksPortfolioSDM()
    sdm_path = "a/b/c"

    # execute
    sdm.load_stored_sdm(path=sdm_path, ioctx=ioctx_mock)

    # assert
    ioctx_mock.get_fns.assert_has_calls(
        [call(sdm_path + "/*.parquet"), call(sdm_path + "/values.json")]
    )
    load_tables_mock.assert_called_once_with(file_path)
    load_values_mock.assert_called_once_with(values_path)
    pd.testing.assert_frame_equal(sdm.sdm_data["file"], df)


@patch("template_pipelines.steps.optimization.preprocessing.sdm.save_tables")
@patch("template_pipelines.steps.optimization.preprocessing.sdm.save_values")
def test_save(save_values_mock, save_tables_mock):
    """Test the saving logic for SDM."""
    # prepare
    sdm = StocksPortfolioSDM()
    sales_df = pd.DataFrame.from_dict({"a": [0]})
    sdm.sdm_data = {"sales": sales_df, "param1": 1}
    ioctx_mock = Mock()
    ioctx_output_path = "ioctx/sdm"
    ioctx_mock.get_output_fn.return_value = ioctx_output_path
    sdm_target_path = "target/sdm"

    # execute
    sdm.save(path=sdm_target_path, ioctx=ioctx_mock)

    # assert
    save_tables_mock.assert_called_once_with({"sales": sales_df}, ioctx_output_path)
    save_values_mock.assert_called_once_with({"param1": 1}, ioctx_output_path)


@patch("template_pipelines.steps.optimization.preprocessing.sdm.load_tables")
def test_create(load_tables_mock):
    """Test that SDM is created correctly."""
    # prepare
    raw_data, result_portfolio_df = get_sdm_io_tables()
    load_tables_mock.return_value = (
        raw_data,
        {
            "industries": "ioctx/path/industries.parquet",
            "stocks": "ioctx/path/stocks.parquet",
            "regions": "ioctx/path/regions.parquet",
            "general_inputs": "ioctx/path/general_inputs.parquet",
        },
    )

    ioctx = Mock()
    sdm = StocksPortfolioSDM()  # sdm instance with mocked I/O components

    # execute
    metrics = sdm.create(data_path="raw/data/path", ioctx=ioctx)

    # assert
    pd.testing.assert_frame_equal(sdm.sdm_data["portfolio_stocks"], result_portfolio_df)
    assert metrics == {
        "general_inputs_rows": 2,
        "industries_rows": 7,
        "stocks_rows": 20,
        "regions_rows": 3,
    }
