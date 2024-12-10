"""Tests for result aggregation."""

from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from template_pipelines.steps.video_analysis.result_aggregation import ResultAggregation


@pytest.fixture(scope="function")
def fixture_result_aggregation():
    """Fixture for the ResultAggregation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        ra = ResultAggregation()
        yield ra


@patch("template_pipelines.steps.video_analysis.result_aggregation.pd.DataFrame.to_csv")
@patch("template_pipelines.steps.video_analysis.result_aggregation.NamedTemporaryFile")
@patch("template_pipelines.steps.video_analysis.result_aggregation.pd.read_csv")
@patch("template_pipelines.steps.video_analysis.result_aggregation.open", new_callable=mock_open)
def test_result_aggregation_run(
    mock_file, mock_pd_read_csv, mock_named_temporary_file, mock_to_csv, fixture_result_aggregation
):
    """Test the ResultAggregation run method."""
    mock_file().readlines.return_value = [
        "1;product_1;usage_1;pros_1;cons_1;recommend_1",
        "2;product_2;usage_2;pros_2;cons_2;recommend_2",
    ]
    mock_pd_read_csv.return_value = pd.DataFrame(
        {"id": ["1", "2"], "video": ["video_1", "video_2"]}
    )
    temporary_file = NamedTemporaryFile()
    mock_named_temporary_file.return_value = temporary_file
    mock_to_csv.return_value = True

    fixture_result_aggregation.inputs = MagicMock()
    fixture_result_aggregation.outputs = {}
    fixture_result_aggregation.run()

    mock_file().readlines.assert_called_once()
    mock_pd_read_csv.assert_called_once()
    mock_to_csv.assert_called_once_with(temporary_file.name, sep=";")
