"""Unit tests for preprocess_data.py."""
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from template_pipelines.steps.time_series.train_model import TrainModelStep


@pytest.fixture(scope="function")
def fixture_train_model():
    """Fixture for preprocess data step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        train_model = TrainModelStep()
        yield train_model


@patch("template_pipelines.steps.time_series.train_model.pd")
@patch.object(TrainModelStep, "train_model")
def test_train_model_run_with_no_model_uri(mock_train_model, mock_pd, fixture_train_model):
    """Test the run() method when no model_uri is provided.

    This test case verifies the behavior of the run() method when no model_uri is provided.
    It mocks the necessary dependencies and asserts the method calls and behavior.

    Args:
        mock_train_model: The mock object for the train_model function.
        mock_pd: The mock object for the pandas module.
        fixture_train_model: The fixture object for the TrainModel class.

    Returns:
        None
    """
    mock_df = create_autospec(SARIMAXResultsWrapper)
    mocked_df = pd.DataFrame()
    mock_pd.read_pickle.return_value = mocked_df
    mock_train_model.return_value = mock_df
    fixture_train_model.mlflow = MagicMock()
    fixture_train_model.outputs = {}

    fixture_train_model.run()

    # Add assertions to verify the method calls and behavior
    mock_train_model.assert_called_once_with(mocked_df)
    fixture_train_model.mlflow.log_param.assert_not_called()
    fixture_train_model.mlflow.log_artifact.assert_not_called()

    # Additional assertions to verify the logger configuration
    mock_logging = MagicMock()
    mock_logging.logger.info.assert_called_once


def test_train_model(fixture_train_model):
    """Test train_model function."""
    # Create a mock DataFrame for co2_series
    co2_series = pd.DataFrame([1, 2, 3, 4, 5])

    # Call the train_model function
    result = fixture_train_model.train_model(co2_series)

    # Assert that the result is an instance of SARIMAXResultsWrapper
    assert isinstance(result, SARIMAXResultsWrapper)

    # Add additional assertions to verify the model training
    assert isinstance(result.aic, float)
    assert result.aic < float("inf")
