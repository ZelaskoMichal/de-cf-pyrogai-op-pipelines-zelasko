"""Unit tests for template_pipelines/steps/vertex_meta/log_metadata.py."""

from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.vertex_meta.log_metadata import LogMetadata


@pytest.fixture(scope="function")
def fixture_log_metadata():
    """Fixture for LogMetadata step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"), patch(
        "vertexai.init"
    ):
        lm = LogMetadata()
        lm.inputs = {
            "metrics": {"mse": 0.1, "mae": 0.2, "r_squared": 0.9},
            "params": {"coeff_0": 0.5, "coeff_1": 0.6, "intercept": 0.7},
        }
        yield lm


@patch("pandas.read_csv")
@patch("joblib.load")
def test_load_data_and_model(mock_load, mock_read_csv, fixture_log_metadata):
    """Test _load_data_and_model method."""
    df, model = fixture_log_metadata._load_data_and_model()
    mock_read_csv.assert_called_once()
    mock_load.assert_called_once()


@patch("google.cloud.aiplatform.log_metrics")
@patch("google.cloud.aiplatform.log_params")
def test_log_run(mock_log_params, mock_log_metrics, fixture_log_metadata):
    """Test _log_run method."""
    my_run = MagicMock()
    model = MagicMock()
    df = MagicMock()
    fixture_log_metadata._log_run(my_run, model, df)
    mock_log_params.assert_called_once()
    mock_log_metrics.assert_called_once()
    my_run.log_model.assert_called_once()


@patch("template_pipelines.steps.vertex_meta.log_metadata.aiplatform.start_run")
@patch("template_pipelines.steps.vertex_meta.log_metadata.LogMetadata._load_data_and_model")
@patch("template_pipelines.steps.vertex_meta.log_metadata.LogMetadata._log_run")
def test_run(mock_log_run, mock_load_data_and_model, mock_start_run, fixture_log_metadata):
    """Testing run method."""
    # Create mock data frame and model
    mock_df = MagicMock()
    mock_model = MagicMock()

    mock_load_data_and_model.return_value = (mock_df, mock_model)

    # Mock the context manager returned by aiplatform.start_run
    mock_context_manager = MagicMock()
    mock_start_run.return_value = mock_context_manager

    # Mock the run object returned by the context manager
    mock_run = MagicMock()
    mock_context_manager.__enter__.return_value = mock_run

    fixture_log_metadata.run()

    mock_load_data_and_model.assert_called_once()
    mock_start_run.assert_called()
    mock_log_run.assert_called_once_with(mock_run, mock_model, mock_df)
