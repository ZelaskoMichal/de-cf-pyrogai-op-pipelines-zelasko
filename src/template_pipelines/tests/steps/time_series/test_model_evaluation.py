"""Unit tests for preprocess_data.py."""
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest

from template_pipelines.steps.time_series.model_evaluation import ModelEvaluationStep


@pytest.fixture(scope="function")
def fixture_model_evaluation():
    """Fixture for preprocess data step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        model_evaluation = ModelEvaluationStep()
        yield model_evaluation


@patch("template_pipelines.steps.time_series.model_evaluation.pd")
@patch.object(ModelEvaluationStep, "generate_metrics")
@patch.object(ModelEvaluationStep, "save_diagnostics_graph")
@patch.object(ModelEvaluationStep, "save_predict_vs_actual_graph")
def test_model_evaluation_run(
    mock_pd,
    mock_generate_metrics,
    mock_save_diagnostics,
    mock_save_predcit,
    fixture_model_evaluation,
):
    """Test run()."""
    mock_df = create_autospec(pd.DataFrame)
    mock_pd.read_pickle.return_value = mock_df
    fixture_model_evaluation.mlflow = MagicMock()
    fixture_model_evaluation.inputs = {"model_uri": {}}

    fixture_model_evaluation.run()


@patch("template_pipelines.steps.time_series.model_evaluation.os.makedirs")
@patch("template_pipelines.steps.time_series.model_evaluation.plt.savefig")
def test_save_diagnostics_graph(mock_os_makedirs, mock_savefig, fixture_model_evaluation):
    """Test save_diagnostics_graph method."""
    min_model_mock = MagicMock()
    min_model_mock.plot_diagnostics = MagicMock()
    output_dir = "test_output_dir"
    fixture_model_evaluation.mlflow = MagicMock()

    fixture_model_evaluation.save_diagnostics_graph(min_model_mock, output_dir)


def test_generate_metrics(fixture_model_evaluation):
    """Test generate_metrics method."""
    co2_truth = pd.DataFrame([1, 2, 3, 4, 5])
    co2_forecasted = pd.DataFrame([1, 2, 3, 4, 6])
    fixture_model_evaluation.mlflow = MagicMock()

    fixture_model_evaluation.generate_metrics(co2_truth, co2_forecasted)


@patch("template_pipelines.steps.time_series.model_evaluation.plt.savefig")
def test_save_predict_vs_actual_graph(mock_savefig, fixture_model_evaluation):
    """Test save_predict_vs_actual_grap method."""
    co2_series = MagicMock()
    co2_series.__getitem__.return_value.plot = MagicMock()

    pred = MagicMock()
    pred.predicted_mean = MagicMock()
    pred.predicted_mean.plot = MagicMock()
    pred.conf_int = MagicMock(
        return_value=pd.DataFrame(
            {"lower": [0, 1, 2], "upper": [3, 4, 5]},
            index=pd.date_range(start="1990-01-01", periods=3),
        )
    )
    output_dir = "test_output_dir"
    fixture_model_evaluation.mlflow = MagicMock()

    fixture_model_evaluation.save_predict_vs_actual_graph(co2_series, pred, output_dir)

    # Happy path assertion
    mock_savefig.assert_called_once_with(f"{output_dir}/predict_vs_actual.png")
