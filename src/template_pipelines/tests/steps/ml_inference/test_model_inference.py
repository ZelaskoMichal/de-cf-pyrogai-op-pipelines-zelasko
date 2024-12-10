"""Unit tests for model_inference.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.ml_inference.model_inference import ModelInference
from template_pipelines.utils.ml_inference import toolkit


@pytest.fixture(scope="function")
def fixture_model_inference():
    """Fixture for ModelInference step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        mi = ModelInference()
        yield mi


def test_model_inference_calculate_mae(fixture_model_inference):
    """Test for calculate_mae."""
    y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    res = fixture_model_inference.calculate_mae(y, y)

    assert res.shape[0] == y.shape[0]
    assert sum(res >= 0) == y.shape[0]
    np.testing.assert_equal(res, np.array([0, 0]))


def test_model_inference_calculate_threshold(fixture_model_inference):
    """Test for calculate_threshold."""
    values = [1, 2, 3, 4, 5]
    res = fixture_model_inference.calculate_threshold(values, 1)
    expected = np.nanmean(values) + np.nanstd(values)

    np.testing.assert_approx_equal(res, expected)


@patch("template_pipelines.steps.ml_inference.model_inference.torch.load")
@patch("template_pipelines.steps.ml_inference.model_inference.AutoEncoder.load_state_dict")
def test_model_inference_load_model_local(
    mock_load_state_dict, mock_torch_load, fixture_model_inference
):
    """Test for local load_model."""
    mock_torch_load.return_value = None
    fixture_model_inference.inputs = MagicMock()
    fixture_model_inference.platform = "Local"
    n_dim = 10

    anomaly_model = fixture_model_inference.load_model(n_dim)

    assert hasattr(anomaly_model, "encoder")
    assert hasattr(anomaly_model, "decoder")
    assert anomaly_model.encoder[0].in_features == n_dim
    assert anomaly_model.decoder[-1].out_features == n_dim


def test_model_inference_load_model_dbr(fixture_model_inference):
    """Test for DBR load_model."""
    fixture_model_inference.inputs = MagicMock()
    fixture_model_inference.platform = "DBR"
    fixture_model_inference.mlflow = MagicMock()
    fixture_model_inference.mlflow.pytorch.load_model.return_value = toolkit.AutoEncoder(10)
    n_dim = 10

    anomaly_model = fixture_model_inference.load_model(n_dim)

    assert hasattr(anomaly_model, "encoder")
    assert hasattr(anomaly_model, "decoder")
    assert anomaly_model.encoder[0].in_features == n_dim
    assert anomaly_model.decoder[-1].out_features == n_dim


def test_model_inference_detect_anomalies(fixture_model_inference):
    """Test for detect_anomalies."""
    dataset = {"loss": np.array([1, 2, 3, 4, np.nan])}
    thre = 2
    res = fixture_model_inference.detect_anomalies(dataset, thre)

    assert np.count_nonzero(res["prediction"]) == 3


@patch("template_pipelines.steps.ml_inference.model_inference.joblib.load")
def test_model_inference_preprocess_data(mock_joblib_load, fixture_model_inference):
    """Test for preprocess_data."""
    mock_joblib_load.return_value = MagicMock()
    mock_joblib_load.transform.return_value = MagicMock()
    training_data = pd.DataFrame({"feature": [], "target": []})

    fixture_model_inference.config = {"ml_inference": {"target": "target"}}
    fixture_model_inference.inputs = MagicMock()
    res = fixture_model_inference.preprocess_data(training_data)

    assert list(res.keys()) == ["data", "labels", "unredeemed"]
    mock_joblib_load.assert_called_once()
    mock_joblib_load.tranform.called_once()


@patch("template_pipelines.steps.ml_inference.model_inference.plt")
def test_model_inference_visualize_distribution_of_losses(mock_plt, fixture_model_inference):
    """Test for visualize_distribution_of_losses."""
    losses = np.random.rand(10)

    mock_fig, mock_axes = mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    fixture_model_inference.mlflow = MagicMock()

    fixture_model_inference.visualize_distribution_of_losses(losses, 1, 2, 3, "output_dir")

    mock_plt.xlabel.assert_called_with("Loss")
    mock_plt.savefig.assert_called()
    mock_plt.close.assert_called()

    assert mock_axes.hist.called
    assert mock_axes.set_title.called
    assert mock_axes.axvline.call_count == 3


@patch("template_pipelines.steps.ml_inference.model_inference.pd.DataFrame.to_csv")
@patch.object(ModelInference, "visualize_distribution_of_losses")
@patch.object(ModelInference, "detect_anomalies")
@patch.object(ModelInference, "calculate_threshold")
@patch.object(ModelInference, "calculate_mae")
@patch.object(ModelInference, "load_model")
@patch.object(ModelInference, "preprocess_data")
@patch("template_pipelines.steps.ml_inference.model_inference.pd.read_parquet")
def test_model_inference_run(
    mock_pd_read_parquet,
    mock_preprocess_data,
    mock_load_model,
    mock_calculate_mae,
    mock_calculate_threshold,
    mock_detect_anomalies,
    mock_visualize_distribution_of_losses,
    mock_pd_to_csv,
    fixture_model_inference,
):
    """Test for run."""
    fixture_model_inference.inputs = MagicMock()
    fixture_model_inference.run()

    mock_pd_read_parquet.assert_called_once()
    mock_preprocess_data.assert_called_once()
    mock_load_model.assert_called_once()
    mock_calculate_mae.assert_called_once()
    assert mock_calculate_threshold.call_count == 3
    mock_detect_anomalies.assert_called_once()
    mock_visualize_distribution_of_losses.assert_called_once()
    mock_pd_to_csv.assert_called_once()
    mock_pd_to_csv.assert_called_once()
