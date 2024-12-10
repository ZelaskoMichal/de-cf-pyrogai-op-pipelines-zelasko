"""Unit tests for mlflow.py."""
from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.operations.mlflow import MlflowStep


@pytest.fixture(scope="function")
def fixture_mlflow():
    """Fixture for step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        mfs = MlflowStep()
        mfs.platform = "DBR"

        yield mfs


@pytest.mark.skip(reason="Needed to be fixed in future")
@patch("template_pipelines.steps.operations.mlflow.plt")
@patch("template_pipelines.steps.operations.mlflow.NamedTemporaryFile")
def test_mlflow_run(mock_named_temp_file, mock_plt, fixture_mlflow):
    """Test run()."""
    fixture_mlflow.mlflow = MagicMock()
    fixture_mlflow.mlflow_utils = MagicMock()
    fixture_mlflow.outputs = MagicMock()

    fixture_mlflow.run()

    fixture_mlflow.mlflow.log_param.assert_called()
    fixture_mlflow.outputs.__setitem__.assert_called()
