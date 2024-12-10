"""Tests for io_helpers."""
from unittest.mock import MagicMock, mock_open, patch

from template_pipelines.utils.aiapps_base.io_helpers import write_csv_file, write_json_file


def test_write_csv_file():
    """Test write_csv_file."""
    mock_df = MagicMock()
    file_name = "file_name.csv"

    mock_tmp = MagicMock()
    mock_tmp.__enter__.return_value.name = "file.csv"
    mock_tmp.__enter__.return_value.write = MagicMock()

    with patch(
        "template_pipelines.utils.aiapps_base.io_helpers.NamedTemporaryFile", return_value=mock_tmp
    ):
        with patch("builtins.open", mock_open()):
            result = write_csv_file(mock_df, file_name)

            mock_df.to_csv.assert_called_once()

            assert result == file_name


@patch("template_pipelines.utils.aiapps_base.io_helpers.json")
def test_write_json_file(mock_json):
    """Test write_json_file."""
    mock_df = MagicMock()
    file_name = "file_name.json"

    mock_tmp = MagicMock()
    mock_tmp.__enter__.return_value.name = "tmp.json"
    mock_tmp.__enter__.return_value.write = MagicMock()

    with patch(
        "template_pipelines.utils.aiapps_base.io_helpers.NamedTemporaryFile", return_value=mock_tmp
    ):
        with patch("builtins.open", mock_open()):
            result = write_json_file(mock_df, file_name)

            mock_json.dump.assert_called_once()

            assert result == file_name
