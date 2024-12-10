"""Unit tests for template_pipelines/steps/aiapps_dbr/score_data.py."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.aiapps_dbr.score_data import ScoreDataStep


@pytest.fixture(scope="function")
def fixture_score_data_step():
    """Fixture for ScoreDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = ScoreDataStep()
        yield sds


@patch("template_pipelines.steps.aiapps_dbr.score_data.write_json_file")
@patch("template_pipelines.steps.aiapps_dbr.score_data.write_csv_file")
@patch("template_pipelines.steps.aiapps_dbr.score_data.shutil")
@patch("template_pipelines.steps.aiapps_dbr.score_data.pickle")
@patch("template_pipelines.steps.aiapps_dbr.score_data.open")
@patch("template_pipelines.steps.aiapps_dbr.score_data.pd")
def test_score_data_step_run(
    mock_pd,
    mock_open,
    mock_pickle,
    mock_shutil,
    mock_write_csv_file,
    mock_write_json_file,
    fixture_score_data_step,
):
    """Test run."""
    fixture_score_data_step.inputs = MagicMock()
    fixture_score_data_step.outputs = MagicMock()
    fixture_score_data_step.ioctx.get_fn.return_value = MagicMock()
    mock_pickle.load.return_value = MagicMock()

    predictions = [0, 1, 0, 1]
    y = [0, 1, 1, 0]
    mock_result_table = pd.DataFrame(
        {
            "Predicted": predictions,
            "Actual": y,
        }
    )

    with patch(
        "template_pipelines.steps.aiapps_dbr.score_data.score_data",
        return_value=(MagicMock(), MagicMock()),
    ) as mock_score_data, patch(
        "template_pipelines.steps.aiapps_dbr.score_data.generate_results_table",
        return_value=mock_result_table,
    ) as mock_generate_results_table, patch(
        "template_pipelines.steps.aiapps_dbr.score_data.plot_confusion_matrix",
        return_value=MagicMock(),
    ) as mock_plot_confusion_matrix, patch(
        "template_pipelines.steps.aiapps_dbr.score_data.plot_precision_recall_curve",
        return_value=MagicMock(),
    ) as mock_plot_precision_recall_curve, patch(
        "template_pipelines.steps.aiapps_dbr.score_data.plot_roc_curve", return_value=MagicMock()
    ) as mock_plot_roc_curve:
        fixture_score_data_step.run()

    mock_score_data.assert_called_once()
    mock_generate_results_table.assert_called_once()
    mock_plot_confusion_matrix.assert_called_once()
    mock_plot_precision_recall_curve.assert_called_once()
    mock_plot_roc_curve.assert_called_once()
    mock_pd.read_csv.assert_called_once()
    assert fixture_score_data_step.ioctx.get_fn.call_count == 2
    mock_open.assert_called()
    mock_pickle.load.assert_called()
    mock_shutil.copy.assert_called()
    assert mock_write_json_file.call_count == 5
    assert mock_write_csv_file.call_count == 1
    fixture_score_data_step.logger.info.assert_called()
