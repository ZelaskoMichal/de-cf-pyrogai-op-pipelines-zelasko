"""Tests for score_data_step_helpers."""
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from template_pipelines.utils.aiapps_base.score_data_step_helpers import (
    generate_results_table,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    score_data,
)


@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.accuracy_score")
def test_score_data(mock_accuracy_score):
    """Test score_data()."""
    mock_predictions = np.array([0.1, 0.2])
    mock_model = Mock()
    mock_model.predict.return_value = mock_predictions

    mocked_x = np.array([[1, 2], [3, 4]])
    mocked_y = np.array([0, 1])

    mock_accuracy_score.return_value = 0.7

    predictions, accuracy = score_data(mock_model, mocked_x, mocked_y)

    mock_model.predict.assert_called_once_with(mocked_x)
    mock_accuracy_score.assert_called_once_with(mocked_y, mock_predictions)
    assert np.array_equal(predictions, np.array([0.1, 0.2]))
    assert accuracy == 0.7


@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.pd")
def test_generate_results_table(mock_pd):
    """Test generate_results_table()."""
    predictions = [0, 1, 0, 1]
    y = [0, 1, 1, 0]

    mock_result_table = pd.DataFrame(
        {
            "Predicted": predictions,
            "Actual": y,
        }
    )
    mock_pd.DataFrame.return_value = mock_result_table

    result_table = generate_results_table(predictions, y)

    mock_pd.DataFrame.assert_called_once()

    expected_results_table = pd.DataFrame(
        {"Predicted": predictions, "Actual": y, "Validation": [True, True, False, False]}
    )

    pd.testing.assert_frame_equal(result_table, expected_results_table)


@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.confusion_matrix")
def test_plot_confusion_matrix(mock_confusion_matrix):
    """Test plot_confusion_matrix()."""
    mock_cm = np.array([[10, 5], [2, 20]])
    mock_confusion_matrix.return_value = mock_cm

    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1]

    result = plot_confusion_matrix(y_true, y_pred)

    expected_heatmap_data = {
        "type": "heatmap",
        "name": "Confusion Matrix",
        "x": ["Predicted Negative", "Predicted Positive"],
        "y": ["Actual Negative", "Actual Positive"],
        "z": mock_cm.tolist(),
    }

    mock_confusion_matrix.assert_called_once_with(y_true, y_pred)
    assert result == expected_heatmap_data


@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.precision_recall_curve")
def test_plot_precision_recall_curve(mock_precision_recall_curve):
    """Test plot_precision_recall_curve()."""
    mock_precision = np.array([0.7, 0.6, 0.5])
    mock_recall = np.array([0.1, 0.2, 0.3])
    mock_precision_recall_curve.return_value = (mock_precision, mock_recall, None)

    y_true = [0, 1, 0]
    y_score = [0.1, 0.3]

    result = plot_precision_recall_curve(y_true, y_score)
    expected_scatter_data = {
        "type": "scatter",
        "name": "Precision-Recall Curve",
        "x": mock_recall.tolist(),
        "y": mock_precision.tolist(),
    }

    mock_precision_recall_curve.assert_called_once_with(y_true, y_score)
    assert result == expected_scatter_data


@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.auc")
@patch("template_pipelines.utils.aiapps_base.score_data_step_helpers.roc_curve")
def test_plot_roc_curve(mock_roc_curve, mock_auc):
    """Test plot_roc_curve()."""
    mock_fpr = np.array([0.1, 0.2, 0.3])
    mock_tpr = np.array([0.4, 0.5, 0.6])
    mock_roc_curve.return_value = (mock_fpr, mock_tpr, None)

    mock_auc_value = 0.8
    mock_auc.return_value = mock_auc_value

    y_true = [0, 1, 0]
    y_score = [0.1, 0.3]
    result = plot_roc_curve(y_true, y_score)

    expected_scatter_data = {
        "type": "scatter",
        "name": "ROC Curve (AUC = {:.2f})".format(mock_auc_value),
        "x": mock_fpr.tolist(),
        "y": mock_tpr.tolist(),
    }

    mock_roc_curve.assert_called_once_with(y_true, y_score)
    mock_auc.assert_called_once_with(mock_fpr, mock_tpr)
    assert result == expected_scatter_data
