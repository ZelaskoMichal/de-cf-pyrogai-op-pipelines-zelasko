"""Unit tests for opinionated_pipelines/steps/iris_3_fix_data.py."""

from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.ml_observability.iris_3_fix_data import FixDataStep


@pytest.fixture(scope="function")
def fixture_fix_data_step():
    """Fixture for FixDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        fds = FixDataStep()
        yield fds


@patch("template_pipelines.steps.ml_observability.iris_3_fix_data.pd")
def test_fix_data_step_run(mock_pd, fixture_fix_data_step):
    """Test run()."""
    fixture_fix_data_step.ioctx.get_fn.return_value = "/"

    mock_data = MagicMock()
    mock_pd.read_csv.return_value = mock_data

    fixture_fix_data_step.run()

    mock_pd.read_csv.assert_called_with("/")
    mock_data.to_csv.assert_called_once()
    fixture_fix_data_step.ioctx.get_fn.assert_called_once_with("standarised.csv")
