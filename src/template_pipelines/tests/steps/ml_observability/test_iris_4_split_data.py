"""Unit tests for opinionated_pipelines/steps/iris_4_split_data.py."""

from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.ml_observability.iris_4_split_data import SplitDataStep


@pytest.fixture(scope="function")
def fixture_split_data_step():
    """Fixture for SplitDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = SplitDataStep()
        yield sds


@patch("template_pipelines.steps.ml_observability.iris_4_split_data.pd")
@patch("template_pipelines.steps.ml_observability.iris_4_split_data.train_test_split", create=True)
def test_split_data_step_run(mock_split, mock_pd, fixture_split_data_step):
    """Test run."""
    mock_split.return_value = (MagicMock(), MagicMock())

    fixture_split_data_step.run()

    fixture_split_data_step.ioctx.get_fn.assert_called_with("fixed.csv")
    mock_pd.read_csv.assert_called_once()
    assert fixture_split_data_step.ioctx.get_output_fn.call_count == 2
    fixture_split_data_step.logger.info.assert_called()
