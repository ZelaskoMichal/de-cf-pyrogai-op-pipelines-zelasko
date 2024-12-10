"""Unit tests for model_training.py."""
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.quickstart.model_training import ModelTraining


@pytest.fixture(scope="function")
def fixture_model_training():
    """Fixture for model training step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        model_training = ModelTraining()
        model_training.logger = MagicMock()
        model_training.ioctx = MagicMock()
        model_training.mlflow = MagicMock()
        model_training.config = {
            "quickstart": {
                "features": ["feature1", "feature2"],
                "n_estimators": 100,
                "random_state": 42,
            }
        }
        model_training.outputs = {}

        model_training.ioctx.get_fn.side_effect = lambda path: f"mocked/path/{path}"
        model_training.ioctx.get_output_fn.side_effect = lambda filename: pathlib.Path(
            f"mocked/output/{filename}"
        )

        yield model_training


@patch("pandas.read_parquet")
@patch("sklearn.ensemble.RandomForestClassifier", autospec=True)
@patch("os.makedirs")
@patch("pandas.DataFrame.to_parquet")
def test_model_training_run(
    mock_to_parquet,
    mock_makedirs,
    mock_classifier,
    mock_read_parquet,
    fixture_model_training,
):
    """Test run() method of ModelTraining."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "target": [0, 1, 0, 1]})

    mock_read_parquet.return_value = df

    mock_classifier_instance = mock_classifier.return_value
    mock_classifier_instance.fit.return_value = None

    mlinfo_mock = MagicMock()
    fixture_model_training.mlflow.sklearn.log_model.return_value = mlinfo_mock

    fixture_model_training.run()

    assert fixture_model_training.logger.info.call_count == 2
    fixture_model_training.logger.info.assert_any_call("Building and compiling a model...")
    fixture_model_training.logger.info.assert_any_call("The model has been trained and saved")

    mock_read_parquet.assert_called_once_with("mocked/path/data.parquet")

    fixture_model_training.mlflow.log_param.assert_any_call("features", ["feature1", "feature2"])
    fixture_model_training.mlflow.log_param.assert_any_call("n_estimators", 100)
    fixture_model_training.mlflow.log_param.assert_any_call("random_state", 42)

    assert fixture_model_training.outputs["model_uri"] == mlinfo_mock.model_uri

    mock_makedirs.assert_called_once_with(
        pathlib.Path("mocked/output/train_test_data"), exist_ok=True
    )

    mock_to_parquet.assert_any_call(pathlib.Path("mocked/output/train_test_data/x_train.parquet"))
    mock_to_parquet.assert_any_call(pathlib.Path("mocked/output/train_test_data/x_test.parquet"))
    mock_to_parquet.assert_any_call(pathlib.Path("mocked/output/train_test_data/y_train.parquet"))
    mock_to_parquet.assert_any_call(pathlib.Path("mocked/output/train_test_data/y_test.parquet"))
