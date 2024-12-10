"""Unit tests for bq io."""
from unittest.mock import ANY, MagicMock, patch

import pytest

from template_pipelines.steps.bq_io.bq_io import BQIo


@pytest.fixture(scope="function")
def fixture_bqio():
    """Fixture for BQIo step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        bqio = BQIo()
        yield bqio


def test_bqio_run_bq_input_methods(fixture_bqio):
    """Test if bq_input methods are correctly called in the run method."""
    mock_bq_input = MagicMock()
    fixture_bqio.inputs = {"bq_slot": mock_bq_input}

    fixture_bqio.config = {
        "bq_io": {
            "gcp_project": "mock_project",
            "bq_dataset": "mock_dataset",
            "bucket_name": "mock_bucket",
        }
    }

    fixture_bqio.run()

    mock_bq_input.head.assert_called_once_with(query=ANY, n_head=5)
    mock_bq_input.count_distinct.assert_called_once_with(
        col="campaign_id", query=ANY, precise=False
    )
    mock_bq_input.to_pandas.assert_called_once_with(query=ANY)
    mock_bq_input.run_query.assert_called_once_with(query=ANY)
    mock_bq_input.save_to_table.assert_called_once_with(
        query=ANY, table_id=ANY, write_disposition="WRITE_TRUNCATE"
    )
    mock_bq_input.extract_to_gcs.assert_called_once_with(query=ANY, uri=ANY, out_format="CSV")
