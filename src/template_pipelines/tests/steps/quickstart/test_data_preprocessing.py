"""Unit tests for data_preprocessing.py."""
from unittest.mock import patch

import pandas as pd
import pytest

from template_pipelines.steps.quickstart.data_preprocessing import Preprocessing


@pytest.fixture(scope="function")
def fixture_preprocessing():
    """Fixture for preprocessing step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        preprocessing = Preprocessing()

        yield preprocessing


@patch("pandas.read_parquet")
@patch("pandas.DataFrame.to_parquet")
def test_preprocessing_run(mock_to_parquet, mock_read_parquet, fixture_preprocessing):
    """Test run()."""
    fixture_preprocessing.inputs = {"input_data": "path/to/input.parquet"}
    fixture_preprocessing.ioctx.get_output_fn.return_value = "path/to/output/data.parquet"

    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_parquet.return_value = mock_df

    fixture_preprocessing.run()

    assert fixture_preprocessing.logger.info.call_count == 2
    mock_read_parquet.assert_called_once_with("path/to/input.parquet")
    fixture_preprocessing.ioctx.get_output_fn.assert_called_once_with("data.parquet")
    mock_to_parquet.assert_called_once_with("path/to/output/data.parquet")
