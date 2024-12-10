"""Unit tests for log_model.py."""

from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from template_pipelines.steps.realtime_inference.log_model import LogModel


@pytest.fixture(scope="function")
def fixture_log_model():
    """Fixture for LogModel step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        lm = LogModel()
        lm.mlflow = MagicMock()
        lm.outputs = {}
        yield lm


def test_run(fixture_log_model):
    """Test run method of LogModel."""
    with patch("template_pipelines.steps.realtime_inference.log_model.load_iris") as mock_load_iris:
        mock_load_iris.return_value = load_iris()

        with patch(
            "template_pipelines.steps.realtime_inference.log_model.train_test_split"
        ) as mock_train_test_split:
            mock_train_test_split.side_effect = train_test_split

            fixture_log_model.run()

            assert fixture_log_model.mlflow.sklearn.log_model.called
            call_args = fixture_log_model.mlflow.sklearn.log_model.call_args
            assert call_args[1] == {"registered_model_name": "realtime_inference_model"}

            assert "realtime_inference_model" in fixture_log_model.outputs
            model_uri = fixture_log_model.outputs["realtime_inference_model"]
            assert model_uri == fixture_log_model.mlflow.sklearn.log_model.return_value.model_uri
