"""Unit tests for opinionated_pipelines/steps/ml_observability/iris_observability_step.py."""
from unittest.mock import MagicMock, call, patch

import pytest

from template_pipelines.steps.ml_observability.iris_observability_step import ModelObservability


@pytest.fixture(scope="function")
def fixture_observability_step():
    """Fixture for ModelObservability step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        mo = ModelObservability()
        yield mo


@patch("template_pipelines.steps.ml_observability.iris_observability_step.pd")
def test_observability_step_run(mock_pd, fixture_observability_step):
    """Test run."""
    mock_pd.read_parquet.return_value = MagicMock()

    fixture_observability_step.observability_clients = MagicMock()
    fixture_observability_step.run()

    mo_client = fixture_observability_step.observability_clients["Iris_Classifier"]

    assert mock_pd.read_parquet.call_count == 2
    assert mo_client.prepare_data.call_count == 2
    assert mo_client.send_data.call_args_list == [call("train"), call("test")]
