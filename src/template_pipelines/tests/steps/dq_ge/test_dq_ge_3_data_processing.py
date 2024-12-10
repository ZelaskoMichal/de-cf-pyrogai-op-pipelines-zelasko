"""Unittests for dqge."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.dq_ge.dq_ge_3_data_processing import DataProcessingStep


@pytest.fixture(scope="function")
def fixture_data_processing():
    """Fixture for DataProcessingStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        dps = DataProcessingStep()
        yield dps


def test_save_dataframe(fixture_data_processing):
    """Test save_dataframe."""
    dps = fixture_data_processing

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    filename = "test_file"

    dps.save_dataframe(df, filename)

    dps.ioctx.get_output_fn.assert_called_once_with(f"{filename}.csv")
    dps.logger.info.assert_called_with(f"Data saved to {filename}")


def test_process_data(fixture_data_processing):
    """Test process_data."""
    dps = fixture_data_processing

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    df2 = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

    processed_df1 = dps.process_data(df1.copy())
    processed_df2 = dps.process_data(df2.copy())

    # Check standardizing of df1
    for col in ["A", "B", "C"]:
        assert processed_df1[col].mean() == pytest.approx(0, abs=1e-9)
        assert processed_df1[col].std() == pytest.approx(1, abs=1e-9)

    # Check scaling of df2
    for col in ["X", "Y"]:
        assert processed_df2[col].min() == 0
        assert processed_df2[col].max() == 1


def test_run(fixture_data_processing):
    """Test run."""
    dps = fixture_data_processing

    mock_df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    mock_df2 = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
    processed_df1 = dps.process_data(mock_df1.copy())
    processed_df2 = dps.process_data(mock_df2.copy())

    with patch.object(pd, "read_csv", side_effect=[mock_df1, mock_df2]), patch.object(
        dps, "save_dataframe"
    ) as mock_save_dataframe, patch.object(pd.DataFrame, "to_csv", MagicMock()):
        dps.run()

        # Check the read_csv calls
        dps.ioctx.get_fn.assert_any_call("data_set_1.csv")
        dps.ioctx.get_fn.assert_any_call("data_set_2.csv")

        # Check the save_dataframe calls
        save_calls = mock_save_dataframe.call_args_list
        assert len(save_calls) == 2

        # Verify the first save_dataframe call
        first_call_df, first_call_filename = save_calls[0][0]
        assert first_call_filename == "processed_df1"
        pd.testing.assert_frame_equal(first_call_df, processed_df1)

        # Verify the second save_dataframe call
        second_call_df, second_call_filename = save_calls[1][0]
        assert second_call_filename == "processed_df2"
        pd.testing.assert_frame_equal(second_call_df, processed_df2)
