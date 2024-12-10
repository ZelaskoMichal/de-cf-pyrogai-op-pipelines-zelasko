"""Unit tests for opinionated_pipelines/steps/model_training.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from template_pipelines.steps.ml_training.model_training import ModelTraining
from template_pipelines.utils.ml_training import toolkit


class MockAnomalyDetector:
    """Mock the Anomaly Detector class for testing purposes."""

    def __call__(self, x):
        """Mock the behavior of calling the anomaly detector."""
        return x

    def train(self):
        """Mock the training method of the anomaly detector."""
        pass

    def eval(self):  # noqa: A003
        """Mock the evaluation method of the anomaly detector."""
        pass


class MockLoss:
    """Mock the mae loss class for testing purposes."""

    def __call__(self, x, y):
        """Mock the bevaior of calling the mae loss."""
        loss = x - y
        loss.backward = lambda: None
        return loss


@pytest.fixture(scope="function")
def fixture_model_training():
    """Fixture for ModelTraining step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        mt = ModelTraining()
        yield mt


def test_model_training_build_model(fixture_model_training):
    """Test for build_model."""
    n_features = 50
    fixture_model_training.config = {"ml_training": {"learning_rate": 0.01}}
    fixture_model_training.mlflow = MagicMock()
    anomaly_detector, loss_fn, optimizer, scheduler = fixture_model_training.build_model(n_features)

    assert isinstance(anomaly_detector, toolkit.AutoEncoder)
    assert isinstance(loss_fn, torch.nn.L1Loss)
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


def test_model_training_train_per_epoch(fixture_model_training):
    """Test for train_per_epoch."""
    x = torch.tensor([50], dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(x)
    loss_value = x - x

    anomaly_detector = MockAnomalyDetector()
    loss_fn = MockLoss()
    optimizer = MagicMock()

    train_loss = fixture_model_training.train_per_epoch(
        dataloader, anomaly_detector, loss_fn, optimizer
    )

    assert train_loss == loss_value


def test_model_training_test_per_epoch(fixture_model_training):
    """Test for test_per_epoch."""
    x = torch.tensor([50], dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(x)
    loss_value = x - x

    anomaly_detector = MockAnomalyDetector()
    loss_fn = MockLoss()

    train_loss = fixture_model_training.test_per_epoch(dataloader, anomaly_detector, loss_fn)

    assert train_loss == loss_value


@patch("template_pipelines.steps.ml_training.model_training.torch.save")
@patch("template_pipelines.steps.ml_training.model_training.plt")
@patch("template_pipelines.steps.ml_training.model_training.os")
@patch("template_pipelines.steps.ml_training.model_training.build_dataloader")
def test_model_training_run_model_training(
    mock_build_dataloader, mock_os, mock_plt, mock_torch_save, fixture_model_training
):
    """Test for run_model_training."""
    mock_build_dataloader.return_value = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    mock_torch_save.return_value = MagicMock()

    fixture_model_training.config = {
        "ml_training": {"epochs": 10, "batch_size": 512, "stop_learning": 5}
    }
    fixture_model_training.train_per_epoch = lambda *args: 0
    fixture_model_training.test_per_epoch = lambda *args: 0
    fixture_model_training.outputs = {"model_uri": {}}

    dataset = {
        "train_unredeemed": pd.DataFrame(),
        "test_unredeemed": pd.DataFrame(),
    }

    fixture_model_training.mlflow = MagicMock()
    anomaly_detector = MagicMock()
    loss_fn = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()

    fixture_model_training.run_model_training(
        anomaly_detector, dataset, loss_fn, optimizer, scheduler
    )

    mock_os.makedirs.assert_called()
    mock_plt.subplots.assert_called()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with("epoch")
    mock_plt.ylabel.assert_called_with("loss")
    mock_plt.legend.assert_called()
    mock_plt.close.assert_called()
    mock_plt.savefig.assert_called()

    fixture_model_training.mlflow.log_artifact.assert_called()


@patch("template_pipelines.steps.ml_training.model_training.torch")
@patch("template_pipelines.steps.ml_training.model_training.np")
@patch("template_pipelines.steps.ml_training.model_training.random")
@patch("template_pipelines.steps.ml_training.model_training.os")
@patch.object(ModelTraining, "run_model_training")
@patch.object(ModelTraining, "build_model")
@patch("template_pipelines.steps.ml_training.model_training.load_tables")
def test_model_training_run(
    mock_load_tables,
    mock_build_model,
    mock_run_model_training,
    mock_os,
    mock_random,
    mock_np,
    mock_torch,
    fixture_model_training,
):
    """Test run."""
    mock_load_tables.return_value = {"train_unredeemed": pd.DataFrame()}
    mock_build_model.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    fixture_model_training.mlflow = MagicMock()

    fixture_model_training.run()

    assert mock_build_model.called
    assert mock_run_model_training.called
    assert fixture_model_training.logger.info.called
    assert mock_os.environ.__setitem__.called
    assert mock_random.seed.called
    assert mock_np.random.seed.called
    assert mock_torch.manual_seed.called
