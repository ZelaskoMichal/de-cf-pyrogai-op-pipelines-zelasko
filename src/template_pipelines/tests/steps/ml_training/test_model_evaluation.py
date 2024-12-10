"""Unit tests for opinionated_pipelines/steps/model_evaluation.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.ml_training.model_evaluation import ModelEvaluation


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


@pytest.fixture(scope="function")
def fixture_model_evaluation():
    """Fixture for ModelEvaluation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        me = ModelEvaluation()
        me.mlflow = MagicMock()
        yield me


def test_model_evaluation_calculate_mae(fixture_model_evaluation):
    """Test for calculate_mae."""
    y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    res = fixture_model_evaluation.calculate_mae(y, y)

    assert res.shape[0] == y.shape[0]
    assert sum(res >= 0) == y.shape[0]
    np.testing.assert_equal(res, np.array([0, 0]))


def test_model_evaluation_convert_labels_df_to_series(fixture_model_evaluation):
    """Test for convert_labels_df_to_series."""
    dataset = {
        "test_labels": pd.DataFrame([1, 2, 3]),
        "train_labels": pd.DataFrame([4, 5, 6]),
    }

    fixture_model_evaluation.convert_labels_df_to_series(dataset)

    assert isinstance(dataset["test_labels"], pd.Series)
    assert isinstance(dataset["train_labels"], pd.Series)

    assert dataset["test_labels"].equals(pd.Series([1, 2, 3]))
    assert dataset["train_labels"].equals(pd.Series([4, 5, 6]))


def test_model_evaluation_create_directory_to_store_model_evaluation(
    fixture_model_evaluation,
):
    """Test for create_directory_to_store_model_evaluation."""
    with patch("template_pipelines.steps.ml_training.model_evaluation.os") as mock_os:
        fixture_model_evaluation.ioctx.get_output_fn.return_value = "output_dir"

        fixture_model_evaluation.create_directory_to_store_model_evaluation()

        mock_os.makedirs.assert_called_once_with("output_dir", exist_ok=True)


@patch("template_pipelines.steps.ml_training.model_evaluation.plt")
@patch("template_pipelines.steps.ml_training.model_evaluation.os.path.join")
def test_model_evaluation_visualize_feature_reconstructions_and_compare(
    mock_os_join, mock_plt, fixture_model_evaluation
):
    """Test for visualize_feature_reconstructions_and_compare."""
    anomaly_detector = MockAnomalyDetector()
    dataset = {
        "test_unredeemed": pd.DataFrame(np.random.rand(10, 10)),
        "test_redeemed": pd.DataFrame(np.random.rand(10, 10)),
    }

    mock_fig, mock_axes = mock_plt.subplots.return_value = (MagicMock(), [MagicMock()])

    fixture_model_evaluation.visualize_feature_reconstructions_and_compare(
        dataset, 10, "output_dir", anomaly_detector
    )

    mock_os_join.assert_any_call("output_dir", "feature_reconstructions.png")
    mock_plt.legend.assert_called()
    mock_plt.savefig.assert_called()
    mock_plt.close.assert_called()

    assert mock_axes[0].plot.call_count == 2
    assert mock_axes[0].fill_between.called
    assert mock_axes[0].set_title.called


@patch("template_pipelines.steps.ml_training.model_evaluation.plt")
@patch("template_pipelines.steps.ml_training.model_evaluation.os.path")
def test_model_evaluation_visualize_distribution_of_losses(
    mock_os, mock_plt, fixture_model_evaluation
):
    """Test for visualize_distribution_of_losses."""
    anomaly_detector = MockAnomalyDetector()
    dataset = {
        "train_unredeemed": pd.DataFrame(np.random.rand(10, 10)),
        "test_redeemed": pd.DataFrame(np.random.rand(10, 10)),
        "test_data": pd.DataFrame(np.random.rand(10, 10)),
    }

    mock_fig, mock_axes = mock_plt.subplots.return_value = (MagicMock(), [MagicMock()])

    fixture_model_evaluation.visualize_distribution_of_losses(
        dataset, "output_dir", anomaly_detector
    )

    mock_plt.xlabel.assert_called_with("Loss")
    mock_plt.savefig.assert_called()
    mock_plt.close.assert_called()

    assert mock_axes[0].hist.called
    assert mock_axes[0].set_title.called
    assert mock_axes[0].axvline.call_count == 3


@patch("template_pipelines.steps.ml_training.model_evaluation.recall_score")
@patch("template_pipelines.steps.ml_training.model_evaluation.precision_score")
@patch("template_pipelines.steps.ml_training.model_evaluation.accuracy_score")
@patch("template_pipelines.steps.ml_training.model_evaluation.confusion_matrix")
def test_model_evaluation_estimate_model_performace_stats(
    mock_confusion_matrix,
    mock_accuracy_score,
    mocck_precision_score,
    mock_recall_score,
    fixture_model_evaluation,
):
    """Test for estimate_model_performace_stats."""
    train_unredeemed_losses = pd.Series([1, 2, 3])
    test_losses = pd.Series([7, 8, 9])
    dataset = MagicMock()
    mock_confusion_matrix.return_value.ravel.return_value = (50, 10, 5, 100)

    res = fixture_model_evaluation.estimate_model_performace_stats(
        train_unredeemed_losses, test_losses, dataset
    )

    assert mock_confusion_matrix.called
    assert mock_accuracy_score.called
    assert mocck_precision_score.called
    assert mock_recall_score.called
    assert isinstance(res, pd.DataFrame)


def test_model_evaluation_select_a_threshold_based(fixture_model_evaluation):
    """Test for select_a_threshold_based."""
    anomaly_detector = MockAnomalyDetector()
    dataset = {"train_data": pd.DataFrame(np.random.rand(10, 3))}

    evaluation = pd.DataFrame({"tp": [5, 10, 15, 20], "std_scalar": [1, 2, 3, 4]})
    train_unredeemed_losses = np.random.rand(10)
    fixture_model_evaluation.config = {"ml_training": {"min_tp": 1}}

    (
        train_losses,
        final_threshold,
        final_metrics,
    ) = fixture_model_evaluation.select_a_threshold_based(
        anomaly_detector, dataset, evaluation, train_unredeemed_losses
    )

    assert isinstance(train_losses, pd.Series)
    assert isinstance(final_threshold, float)
    assert isinstance(final_metrics, pd.Series)


@patch("matplotlib.pyplot.close")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.title")
@patch("template_pipelines.steps.ml_training.model_evaluation.np.less")
@patch("template_pipelines.steps.ml_training.model_evaluation.os.path.join")
@patch("template_pipelines.steps.ml_training.model_evaluation.pd.DataFrame.to_parquet")
def test_get_and_store_predictions(
    mock_to_parquet,
    mock_path_join,
    mock_np_less,
    mock_plt_title,
    mock_plt_savefig,
    mock_plt_close,
    fixture_model_evaluation,
):
    """Test get_and_store_predictions."""
    train_losses = MagicMock()
    final_threshold = MagicMock()
    dataset = {
        "train_data": pd.DataFrame(),
        "train_labels": pd.Series([True, False, True]),
    }
    output_dir = MagicMock()
    final_metrics = pd.DataFrame({"tn": [1], "fp": [0], "fn": [0], "tp": [1]})

    mock_np_less.return_value.numpy.return_value = 1
    mock_path_join.return_value = "mo_train_data.parquet"

    fixture_model_evaluation.get_and_store_predictions(
        train_losses, final_threshold, dataset, output_dir, final_metrics
    )

    mock_path_join.assert_any_call(output_dir, "mo_train_data.parquet")
    mock_path_join.assert_any_call(output_dir, "confusion_matrix.png")

    mock_to_parquet.assert_called_once_with("mo_train_data.parquet")

    mock_plt_title.assert_called_once_with("Confusion Matrix")
    mock_plt_savefig.assert_called_once_with("mo_train_data.parquet")
    mock_plt_close.assert_called()


@patch.object(ModelEvaluation, "get_and_store_predictions")
@patch.object(ModelEvaluation, "select_a_threshold_based")
@patch.object(ModelEvaluation, "estimate_model_performace_stats")
@patch.object(ModelEvaluation, "visualize_distribution_of_losses")
@patch.object(ModelEvaluation, "visualize_feature_reconstructions_and_compare")
@patch.object(ModelEvaluation, "create_directory_to_store_model_evaluation")
@patch.object(ModelEvaluation, "convert_labels_df_to_series")
@patch("template_pipelines.steps.ml_training.model_evaluation.load_tables")
def test_model_evaluation_run(
    mock_load_tables,
    mock_convert_labels_df_to_series,
    mock_create_directory_to_store_model_evaluation,
    mock_visualize_feature_reconstructions_and_compare,
    mock_visualize_distribution_of_losses,
    mock_estimate_model_performace_stats,
    mock_select_a_threshold_based,
    mock_get_and_store_predictions,
    fixture_model_evaluation,
):
    """Test for run."""
    fixture_model_evaluation.inputs = {"model_uri": "model_uri"}
    mock_visualize_distribution_of_losses.return_value = (MagicMock(), MagicMock())
    mock_select_a_threshold_based.return_value = (MagicMock(), MagicMock(), MagicMock())

    fixture_model_evaluation.run()

    assert mock_load_tables.called
    assert mock_convert_labels_df_to_series.called
    assert mock_create_directory_to_store_model_evaluation.called
    assert mock_visualize_feature_reconstructions_and_compare.called
    assert mock_visualize_distribution_of_losses.called
    assert mock_estimate_model_performace_stats.called
    assert mock_select_a_threshold_based.called
    assert mock_select_a_threshold_based.called
    assert mock_get_and_store_predictions.called
