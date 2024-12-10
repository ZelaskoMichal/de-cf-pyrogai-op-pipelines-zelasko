"""Unit tests for log_model.py."""

from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from template_pipelines.steps.mdf_iiot.log_model import LogModel


@pytest.fixture(scope="function")
def fixture_log_model():
    """Fixture for LogModel step."""
    with patch("aif.pyrogai.steps.step.PlatformProvider"):
        lm = LogModel()
        lm.mlflow = MagicMock()
        lm.config = {
            "model_name": "mdf_model",
            "model_flavor": "sklearn",
            "model_type": "categorical",
        }
        lm.outputs = {}
        yield lm


def test_run(fixture_log_model):
    """Test run method of LogModel."""
    with patch("template_pipelines.steps.mdf_iiot.log_model.load_iris") as mock_load_iris:
        mock_load_iris.return_value = load_iris()

        with patch(
            "template_pipelines.steps.mdf_iiot.log_model.train_test_split"
        ) as mock_train_test_split:
            mock_train_test_split.side_effect = train_test_split

            fixture_log_model.run()

            # Verify that mlflow.log_model_with_tags was called
            assert fixture_log_model.mlflow.log_model_with_tags.called
            call_args = fixture_log_model.mlflow.log_model_with_tags.call_args

            # Verify the arguments passed to log_model_with_tags
            assert call_args[1]["classifier"] is not None
            assert call_args[1]["model_name"] == "mdf_model"
            assert call_args[1]["flavor"] == "sklearn"
            assert call_args[1]["tags"] == {"model_type": "categorical"}

            # Verify the outputs
            assert "mdf_model_uri" in fixture_log_model.outputs
            model_uri = fixture_log_model.outputs["mdf_model_uri"]
            assert model_uri == fixture_log_model.mlflow.log_model_with_tags.return_value.model_uri
