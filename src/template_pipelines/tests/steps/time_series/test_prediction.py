"""Unit tests for preprocess_data.py."""
import os
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest

from template_pipelines.steps.time_series.prediction import PredictionStep


@pytest.fixture(scope="function")
def fixture_prediction():
    """Fixture for preprocess data step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        prediction = PredictionStep()
        yield prediction


@patch("template_pipelines.steps.time_series.prediction.pd")
@patch.object(PredictionStep, "save_forecast_graph")
def test_model_evaluation_run(
    mock_pd,
    mock_save_forecast_graph,
    fixture_prediction,
):
    """Test run()."""
    fixture_prediction.ioctx.get_output_fn.return_value = "/"

    fixture_prediction.inputs = {"model_uri": {}}
    fixture_prediction.mlflow = MagicMock()

    mock_df = create_autospec(pd.DataFrame)
    mock_pd.read_pickle.return_value = mock_df

    fixture_prediction.run()


@patch("template_pipelines.steps.time_series.prediction.plt.savefig")
def test_save_forecast_graph(mock_savefig, fixture_prediction):
    """Test the save_forecast_graph function.

    This function tests the functionality of the save_forecast_graph method in the Prediction class.
    It verifies that the method correctly saves the forecast graph and performs the necessary assertions.

    Args:
    - mock_savefig: A MagicMock object representing the savefig function.
    - fixture_prediction: A MagicMock object representing the fixture for the Prediction class.

    Returns:
    None

    Raises:
    AssertionError: If any of the assertions fail.
    """
    min_model = MagicMock()
    pred_uc = MagicMock()
    pred_uc.predicted_mean = MagicMock()
    pred_uc.conf_int = MagicMock(
        return_value=pd.DataFrame(
            {"lower": [0, 1, 2], "upper": [3, 4, 5]},
            index=pd.date_range(start="2022-01-01", periods=3),
        )
    )
    min_model.get_forecast.return_value = pred_uc
    co2_series = MagicMock()
    co2_series.plot = MagicMock()
    output_dir = "test_output_dir"
    output_path = os.path.join(output_dir, "forecast.png")
    fixture_prediction.mlflow = MagicMock()
    fixture_prediction.save_forecast_graph(
        min_model, co2_series, output_dir
    )  # Corrected method name

    # Assertions for happy path
    mock_savefig.assert_called_once()
    mock_savefig.assert_called_with(output_path)
    fixture_prediction.mlflow.log_artifact.assert_called_once()
    fixture_prediction.mlflow.log_artifact.assert_called_with(output_path, artifact_path="graphs")
    co2_series.plot.assert_called_once  # test the plot method
    # test axis labels
    co2_series.plot.assert_called_with(label="observed", figsize=(15, 8))
    pred_uc.predicted_mean.plot.assert_called_once

    # Assertions for unhappy path
    assert min_model.get_forecast.call_count == 1
    assert co2_series.plot.call_count == 1
    assert fixture_prediction.mlflow.log_artifact.call_count == 1
