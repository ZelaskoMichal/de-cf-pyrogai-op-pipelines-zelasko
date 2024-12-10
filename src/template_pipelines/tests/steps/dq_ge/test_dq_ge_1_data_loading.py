"""Unittests for dqge."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.dq_ge.dq_ge_1_data_loading import DataLoaderStep


@pytest.fixture(scope="function")
def fixture_data_loader():
    """Fixture for DataLoaderStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        dls = DataLoaderStep()
        yield dls


def test_generate_random_dataframe(fixture_data_loader):
    """Test generate_random_dataframe."""
    dls = fixture_data_loader

    df = dls.generate_random_dataframe(num_rows=10, columns=["A", "B"])
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 2)
    assert list(df.columns) == ["A", "B"]


def test_save_dataframe(fixture_data_loader):
    """Test save_dataframe."""
    dls = fixture_data_loader

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    filename = "test_file"

    dls.save_dataframe(df, filename)

    dls.ioctx.get_output_fn.assert_called_once_with(f"{filename}.csv")
    dls.logger.info.assert_called_with(f"Data saved to {filename}")


def test_run(fixture_data_loader):
    """Test run."""
    dls = fixture_data_loader

    with patch.object(
        dls, "generate_random_dataframe"
    ) as mock_generate_random_dataframe, patch.object(
        dls, "save_dataframe"
    ) as mock_save_dataframe, patch.object(
        pd.DataFrame, "to_csv", MagicMock()
    ):
        mock_df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        mock_df2 = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

        mock_generate_random_dataframe.side_effect = [mock_df1, mock_df2]

        dls.run()

        dls.logger.info.assert_any_call("Start data loading step")
        mock_generate_random_dataframe.assert_any_call(num_rows=150, columns=["A", "B", "C"])
        mock_generate_random_dataframe.assert_any_call(num_rows=200, columns=["X", "Y"])

        # Check the save_dataframe calls
        save_calls = mock_save_dataframe.call_args_list
        assert len(save_calls) == 2

        # Verify the first save_dataframe call
        first_call_df, first_call_filename = save_calls[0][0]
        assert first_call_filename == "data_set_1"
        assert first_call_df.equals(mock_df1)

        # Verify the second save_dataframe call
        second_call_df, second_call_filename = save_calls[1][0]
        assert second_call_filename == "data_set_2"
        assert second_call_df.equals(mock_df2)

        dls.logger.info.assert_any_call("Finish data loading step")
