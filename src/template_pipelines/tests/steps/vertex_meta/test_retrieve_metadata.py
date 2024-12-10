"""Unit tests for template_pipelines/steps/vertex_meta/retrieve_metadata.py."""

from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.vertex_meta.retrieve_metadata import RetrieveMetadata


@pytest.fixture(scope="function")
def fixture_retrieve_metadata():
    """Fixture for RetrieveMetadata step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"), patch(
        "vertexai.init"
    ):
        rm = RetrieveMetadata()
        yield rm


@patch("google.cloud.aiplatform.ExperimentRun")
def test_get_experiment_run(mock_experiment_run, fixture_retrieve_metadata):
    """Tests _get_experiment_run method."""
    fixture_retrieve_metadata._get_experiment_run()
    mock_experiment_run.assert_called_once()


@patch("google.cloud.aiplatform.ExperimentRun.get_metrics")
@patch("google.cloud.aiplatform.ExperimentRun.get_params")
@patch("google.cloud.aiplatform.ExperimentRun.get_experiment_models")
def test_get_experiment_data(
    mock_get_experiment_models, mock_get_params, mock_get_metrics, fixture_retrieve_metadata
):
    """Tests _get_experiment_data method."""
    experiment_run = MagicMock()
    experiment_run.get_metrics = mock_get_metrics
    experiment_run.get_params = mock_get_params
    experiment_run.get_experiment_models = mock_get_experiment_models
    metrics, parameters, model_list, loaded_model = fixture_retrieve_metadata._get_experiment_data(
        experiment_run
    )
    mock_get_metrics.assert_called_once()
    mock_get_params.assert_called_once()
    mock_get_experiment_models.assert_called_once()


@patch(
    "template_pipelines.steps.vertex_meta.retrieve_metadata.RetrieveMetadata._get_experiment_run"
)
@patch(
    "template_pipelines.steps.vertex_meta.retrieve_metadata.RetrieveMetadata._get_experiment_data"
)
def test_run(mock_get_experiment_data, mock_get_experiment_run, fixture_retrieve_metadata):
    """Testing run method."""
    # Create mock experiment run and data
    mock_experiment_run = MagicMock()
    mock_experiment_data = ("metrics", "parameters", ["model1", "model2"], "loaded_model")

    mock_get_experiment_run.return_value = mock_experiment_run
    mock_get_experiment_data.return_value = mock_experiment_data

    fixture_retrieve_metadata.run()

    mock_get_experiment_run.assert_called_once()
    mock_get_experiment_data.assert_called_once_with(mock_experiment_run)
