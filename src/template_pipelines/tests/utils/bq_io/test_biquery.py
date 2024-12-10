"""Tests for BigQuery Class."""
from unittest.mock import MagicMock, patch

import pytest
from google.cloud import bigquery

from template_pipelines.utils.bq_io.bigquery import BigQuery


@pytest.fixture(scope="function")
def fixture_big_query():
    """Fixture for BigQuery."""
    with patch("template_pipelines.utils.bq_io.bigquery.bigquery"):
        with patch("template_pipelines.utils.bq_io.bigquery.SchemaField"):
            bq = BigQuery()

            yield bq


def test_big_query_to_pandas(fixture_big_query):
    """Test to_pandas."""
    mock_query = MagicMock()
    mock_query.to_dataframe.return_value = MagicMock()
    fixture_big_query.client.query.return_value = mock_query

    fixture_big_query.to_pandas("query")

    fixture_big_query.client.query.assert_called_once_with("query")
    mock_query.to_dataframe.assert_called_once()


def test_big_query_run_query(fixture_big_query):
    """Test run_query."""
    mock_query = MagicMock()
    mock_query.result.return_value = MagicMock()
    fixture_big_query.client.query.return_value = mock_query

    fixture_big_query.run_query("query")

    fixture_big_query.client.query.assert_called_once_with("query")
    mock_query.result.assert_called_once()


def test_big_query_save_to_table(fixture_big_query):
    """Test save_to_table."""
    mock_query = MagicMock()
    mock_query.result.return_value = MagicMock()
    fixture_big_query.client.query.return_value = mock_query

    with patch("template_pipelines.utils.bq_io.bigquery.bigquery.QueryJobConfig") as mock_config:
        mock_config.return_value = MagicMock()
        fixture_big_query.save_to_table("query", "table_id", "WRITE_EMPTY")

        mock_config.assert_called_once_with(destination="table_id", write_disposition="WRITE_EMPTY")
        fixture_big_query.client.query.assert_called_once_with(
            "query", job_config=mock_config.return_value
        )
        mock_query.result.assert_called_once()


def test_big_query_extract_to_gcs(fixture_big_query):
    """Test extract_to_gcs."""
    mock_client = MagicMock()
    fixture_big_query.client = mock_client

    with patch("template_pipelines.utils.bq_io.bigquery.bigquery.ExtractJobConfig") as mock_config:
        fixture_big_query.extract_to_gcs("query", "uri", bigquery.DestinationFormat.CSV)

        mock_client.query.assert_called_once_with("query")
        mock_client.extract_table.assert_called_once()
        mock_config.assert_called_once()


def test_big_query_create_temp_table(fixture_big_query):
    """Test create_temp_table."""
    mock_table = MagicMock()
    mock_table.project = "test_project"
    mock_table.dataset_id = "test_dataset"
    mock_table.table_id = "test_table_id"

    fixture_big_query.client.create_table.return_value = mock_table
    fixture_big_query.client.delete_table = MagicMock()

    mock_schema_field = MagicMock()
    mock_schema_field.side_effect = [
        bigquery.SchemaField("column1", "STRING", "REQUIRED"),
        bigquery.SchemaField("column2", "STRING", "NULLABLE"),
    ]
    with patch(
        "template_pipelines.utils.bq_io.bigquery.bigquery.SchemaField",
        new=mock_schema_field,
    ):
        table_location = fixture_big_query.create_temp_table(
            "test_project.test_dataset.test_table_id",
            [("column1", "STRING", "REQUIRED"), ("column2", "STRING", "NULLABLE")],
            expires_in_hours=1,
        )

    expected_table_location = "test_project.test_dataset.test_table_id"
    assert table_location == expected_table_location

    assert mock_schema_field.call_count == 2

    fixture_big_query.client.create_table.assert_called_once()
    fixture_big_query.client.delete_table.assert_called_once_with(
        "test_project.test_dataset.test_table_id", not_found_ok=True
    )


def test_big_query_insert_rows(fixture_big_query):
    """Test insert_rows in BigQuery."""
    mock_table = MagicMock()
    mock_data = [{"column1": "value1", "column2": "value2"}]
    fixture_big_query.client.get_table.return_value = mock_table
    fixture_big_query.client.insert_rows.return_value = []

    fixture_big_query.insert_rows("your-project.your_dataset.your_table", mock_data)

    fixture_big_query.client.get_table.assert_called_once_with(
        "your-project.your_dataset.your_table"
    )
    fixture_big_query.client.insert_rows.assert_called_once_with(mock_table, mock_data)


def test_big_query_insert_rows_with_error(fixture_big_query):
    """Test insert_rows with error."""
    mock_table = MagicMock()
    mock_data = [{"column1": "value1", "column2": "value2"}]
    fixture_big_query.client.get_table.return_value = mock_table
    fixture_big_query.client.insert_rows.return_value = [{"error": "test error"}]

    with pytest.raises(ValueError):
        fixture_big_query.insert_rows("your-project.your_dataset.your_table", mock_data)

    fixture_big_query.client.get_table.assert_called_once_with(
        "your-project.your_dataset.your_table"
    )
    fixture_big_query.client.insert_rows.assert_called_once_with(mock_table, mock_data)


def test_big_query_delete_table(fixture_big_query):
    """Test delete_table."""
    fixture_big_query.client.delete_table = MagicMock()

    fixture_big_query.delete_table("your-project.your_dataset.your_table")

    fixture_big_query.client.delete_table.assert_called_once_with(
        "your-project.your_dataset.your_table", not_found_ok=True
    )


def test_big_query_create_table(fixture_big_query):
    """Test create_table."""
    mock_schema = {"field1": "STRING", "field2": "INT64"}
    mock_table = MagicMock()

    fixture_big_query.client.create_table.return_value = mock_table

    with patch("template_pipelines.utils.bq_io.bigquery.bigquery.SchemaField", new=MagicMock()):
        fixture_big_query.create_table("your-project.your_dataset.your_table", mock_schema)

    fixture_big_query.client.create_table.assert_called_once()
