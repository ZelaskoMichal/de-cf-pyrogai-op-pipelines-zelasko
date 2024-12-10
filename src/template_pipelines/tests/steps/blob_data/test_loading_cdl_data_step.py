"""Test loading Blob data step."""

from unittest.mock import MagicMock, patch

import pytest

from src.template_pipelines.steps.blob_data.loading_blob_data_step import (  # noqa
    LoadingBlobDataStep,
)


@pytest.fixture(scope="function")
def fixture_loading_blob_data_step():
    """Fixture for LoadingBlobDataStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = LoadingBlobDataStep()
        yield step


@patch("os.walk")
@patch("pyarrow.parquet.read_table")
def test_combine_parquet_files(mock_read_table, mock_os_walk):
    """Test combining Parquet files."""
    # Setup mock for os.walk
    mock_os_walk.return_value = [
        ("/fake_dir", ("subdir",), ("file1.snappy.parquet", "file2.snappy.parquet")),
        ("/fake_dir/subdir", (), ("file3.snappy.parquet",)),
    ]

    # Setup mock for pyarrow.parquet.read_table
    mock_table1 = MagicMock()
    mock_table2 = MagicMock()
    mock_table3 = MagicMock()
    mock_read_table.side_effect = [mock_table1, mock_table2, mock_table3]

    # Setup mock for pa.concat_tables
    mock_combined_table = MagicMock()
    with patch(
        "pyarrow.concat_tables", return_value=mock_combined_table
    ) as mock_concat_tables, patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = LoadingBlobDataStep()
        step.logger = MagicMock()
        combined_table = step.combine_parquet_files("/fake_dir")

        # Assertions
        assert combined_table == mock_combined_table
        mock_os_walk.assert_called_once_with("/fake_dir")
        assert mock_read_table.call_count == 3
        mock_read_table.assert_any_call("/fake_dir/file1.snappy.parquet")
        mock_read_table.assert_any_call("/fake_dir/file2.snappy.parquet")
        mock_read_table.assert_any_call("/fake_dir/subdir/file3.snappy.parquet")
        mock_concat_tables.assert_called_once_with([mock_table1, mock_table2, mock_table3])
        step.logger.info.assert_any_call(
            "Combining Parquet files from folder and subfolders: /fake_dir"
        )
        step.logger.info.assert_any_call("Found 3 Parquet files")
        step.logger.info.assert_any_call(
            f"Combined Parquet DataFrame. Columns: {mock_combined_table.column_names}"
        )


@patch("pyspark.sql.SparkSession.builder.getOrCreate")
def test_load_from_dbr(mock_get_or_create, fixture_loading_blob_data_step):
    """Test loading data from DBR."""
    mock_df = MagicMock()
    mock_session = mock_get_or_create.return_value
    mock_session.table.return_value = mock_df
    fixture_loading_blob_data_step.runtime_parameters = {"dbr_table": "test_table"}

    df = fixture_loading_blob_data_step.load_from_dbr()

    mock_session.table.assert_called_with("test_table")
    assert df == mock_df


@patch("pyarrow.parquet.read_table")
def test_load_parquet_file(mock_read_table, fixture_loading_blob_data_step):
    """Test loading a single Parquet file."""
    mock_table = MagicMock()
    mock_read_table.return_value = mock_table
    file_path = "test.parquet"

    table = fixture_loading_blob_data_step.load_parquet_file(file_path)

    mock_read_table.assert_called_with(file_path)
    assert table == mock_table


@pytest.mark.parametrize("platform", ["DBR", "AML", "Local"])
@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.load_from_dbr"
)
@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.load_from_aml"
)
@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.load_from_local"
)
def test_run(
    mock_load_from_local,
    mock_load_from_aml,
    mock_load_from_dbr,
    platform,
    fixture_loading_blob_data_step,
):
    """Test the run method for different platforms."""
    fixture_loading_blob_data_step.platform = platform
    mock_df = MagicMock()
    mock_load_from_dbr.return_value = mock_df
    mock_load_from_aml.return_value = mock_df
    mock_load_from_local.return_value = mock_df

    fixture_loading_blob_data_step.run()

    if platform == "DBR":
        mock_load_from_dbr.assert_called_once()
    elif platform == "AML":
        mock_load_from_aml.assert_called_once()
    elif platform == "Local":
        mock_load_from_local.assert_called_once()
    else:
        pytest.fail(f"Unsupported platform: {platform}")


@pytest.mark.parametrize(
    "input_scenario, expected_exception",
    [
        ({"folder_path": "None", "file_path": "None"}, RuntimeError),
        ({"folder_path": "path", "file_path": "path"}, RuntimeError),
    ],
)
def test_load_from_aml_exceptions(
    input_scenario, expected_exception, fixture_loading_blob_data_step
):
    """Test load_from_aml for scenarios that should raise exceptions."""
    fixture_loading_blob_data_step.runtime_parameters = input_scenario

    with pytest.raises(expected_exception):
        fixture_loading_blob_data_step.load_from_aml()


@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.combine_parquet_files"
)
def test_load_from_aml_with_folder_path(mock_combine, fixture_loading_blob_data_step):
    """Test load_from_aml successfully loads data from folder path."""
    test_folder_path = "/path/to/folder"
    fixture_loading_blob_data_step.runtime_parameters = {
        "folder_path": test_folder_path,
        "file_path": "",
    }
    fixture_loading_blob_data_step.inputs = {"blob_folder": test_folder_path}

    fixture_loading_blob_data_step.load_from_aml()

    mock_combine.assert_called_once_with(test_folder_path)


@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.load_parquet_file"
)
def test_load_from_aml_with_file_path(mock_load_parquet, fixture_loading_blob_data_step):
    """Test load_from_aml successfully loads data from file path."""
    test_file_path = "/path/to/file.parquet"
    fixture_loading_blob_data_step.runtime_parameters = {
        "folder_path": "",
        "file_path": test_file_path,
    }
    fixture_loading_blob_data_step.inputs = {"blob_file": test_file_path}

    fixture_loading_blob_data_step.load_from_aml()

    mock_load_parquet.assert_called_once_with(test_file_path)


@patch(
    "src.template_pipelines.steps.blob_data.loading_blob_data_step.LoadingBlobDataStep.load_from_aml"
)
def test_load_from_local(mock_load_from_aml, fixture_loading_blob_data_step):
    """Test load_from_local delegates to load_from_aml."""
    fixture_loading_blob_data_step.load_from_local()
    mock_load_from_aml.assert_called_once()


def test_run_with_invalid_platform(fixture_loading_blob_data_step):
    """Test run method raises RuntimeError for invalid platform."""
    fixture_loading_blob_data_step.platform = "invalid_platform"

    with pytest.raises(RuntimeError) as exc_info:
        fixture_loading_blob_data_step.run()

    assert "Wrong platform selected" in str(exc_info.value)


def test_load_from_dbr_without_dbr_table(fixture_loading_blob_data_step):
    """Test load_from_dbr raises RuntimeError when 'dbr_table' is missing."""
    fixture_loading_blob_data_step.runtime_parameters = {}  # No 'dbr_table' provided.

    with pytest.raises(RuntimeError) as exc_info:
        fixture_loading_blob_data_step.load_from_dbr()

    assert "Lack of 'dbr_table' parameter" in str(exc_info.value)
