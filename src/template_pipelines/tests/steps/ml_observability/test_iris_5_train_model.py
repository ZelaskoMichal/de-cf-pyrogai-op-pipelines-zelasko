"""Unit tests for opinionated_pipelines/steps/iris_5_train_model.py."""
from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.ml_observability.iris_5_train_model import TrainModelStep


@pytest.fixture(scope="function")
def fixture_train_model_step():
    """Fixture for TrainModelStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        tms = TrainModelStep()
        yield tms


@patch("template_pipelines.steps.ml_observability.iris_5_train_model.pd")
@patch("template_pipelines.steps.ml_observability.iris_5_train_model.pickle")
@patch("template_pipelines.steps.ml_observability.iris_5_train_model.open")
@patch("template_pipelines.steps.ml_observability.iris_5_train_model.SVC", create=True)
def test_train_model_step_run(
    mock_classifier, mock_open, mock_pickle, mock_pd, fixture_train_model_step
):
    """Test run."""
    mock_df = MagicMock()
    mock_df.reset_index.return_value = mock_df
    mock_df.__len__.return_value = 1

    mock_pd.read_parquet.return_value = mock_df

    fixture_train_model_step.run()

    mock_pd.read_parquet.assert_called_once()
    assert mock_df.reset_index.called
    assert mock_df.__len__.called
    mock_classifier.assert_called_once()
    mock_df.to_parquet.assert_called_once()
    mock_pickle.dump.assert_called()
    mock_open.assert_called()
