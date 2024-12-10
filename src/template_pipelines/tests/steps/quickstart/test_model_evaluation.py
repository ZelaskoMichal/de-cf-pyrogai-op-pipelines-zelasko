"""Unit tests for model_evaluation.py."""
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from template_pipelines.steps.quickstart.model_evaluation import ModelEvaluation


@pytest.fixture(scope="function")
def fixture_model_evaluation():
    """Fixture for model evaluation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        model_evaluation = ModelEvaluation()
        model_evaluation.mlflow = MagicMock()
        model_evaluation.mlflow_utils = MagicMock()
        model_evaluation.inputs = {"model_uri": "some/model/uri"}

        model_evaluation.ioctx.get_fn.side_effect = lambda path: f"mocked/path/{path}"
        model_evaluation.ioctx.get_output_fn.side_effect = (
            lambda filename: f"mocked/output/{filename}"
        )

        yield model_evaluation


@pytest.mark.skip(reason="Needed to be fixed in future")
@patch("pandas.read_parquet")
@patch("sklearn.ensemble.RandomForestClassifier", autospec=True)
@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.plot")
@patch("matplotlib.pyplot.savefig")
@patch("numpy.argsort")
def test_model_evaluation_run(
    mock_argsort,
    mock_savefig,
    mock_plot,
    mock_figure,
    mock_classifier,
    mock_read_parquet,
    fixture_model_evaluation,
):
    """Test run() method of ModelEvaluation."""
    x_test = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    y_test = pd.Series([0, 1])
    y_pred = np.array([0, 1])
    y_pred_prob = np.array([[0.6, 0.4], [0.4, 0.6]])

    mock_read_parquet.side_effect = [x_test, y_test]
    mock_classifier_instance = mock_classifier.return_value
    mock_classifier_instance.predict.return_value = y_pred
    mock_classifier_instance.predict_proba.return_value = y_pred_prob
    mock_classifier_instance.feature_importances_ = np.array([0.7, 0.3])
    mock_argsort.return_value = np.array([0, 1])

    fixture_model_evaluation.mlflow.sklearn.load_model.return_value = mock_classifier_instance

    fixture_model_evaluation.run()

    assert fixture_model_evaluation.logger.info.call_count == 2

    mock_read_parquet.assert_any_call("mocked/path/train_test_data/x_test.parquet")
    mock_read_parquet.assert_any_call("mocked/path/train_test_data/y_test.parquet")

    fixture_model_evaluation.mlflow.sklearn.load_model.assert_called_once_with("some/model/uri")

    mock_classifier_instance.predict.assert_called_once_with(x_test)
    mock_classifier_instance.predict_proba.assert_called_once_with(x_test)

    fixture_model_evaluation.mlflow.log_metric.assert_any_call(
        "accuracy", accuracy_score(y_test, y_pred)
    )
    fixture_model_evaluation.mlflow.log_metric.assert_any_call(
        "precision", precision_score(y_test, y_pred)
    )
    fixture_model_evaluation.mlflow.log_metric.assert_any_call(
        "recall", recall_score(y_test, y_pred)
    )
    fixture_model_evaluation.mlflow.log_metric.assert_any_call("f1_score", f1_score(y_test, y_pred))
    fixture_model_evaluation.mlflow.log_metric.assert_any_call(
        "roc_auc", roc_auc_score(y_test, y_pred_prob[:, 1])
    )

    fixture_model_evaluation.mlflow_utils.log.assert_called_once_with(
        log_to_root_run=True,
        metrics={"accuracy": accuracy_score(y_test, y_pred)},
    )

    mock_figure.assert_called()
    mock_plot.assert_any_call([0, 1], [0, 1], "k--")
    mock_plot.assert_any_call(mock.ANY, mock.ANY)
    mock_savefig.assert_any_call("mocked/output/feature_importance.png")
    fixture_model_evaluation.mlflow.log_artifact.assert_any_call(
        "mocked/output/feature_importance.png"
    )

    mock_savefig.assert_any_call("mocked/output/feature_importance.png")
    fixture_model_evaluation.mlflow.log_artifact.assert_any_call(
        "mocked/output/feature_importance.png"
    )
