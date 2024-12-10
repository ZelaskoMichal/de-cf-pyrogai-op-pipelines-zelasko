"""Unit tests for template_pipelines/steps/iris_6_score_data.py."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.ml_iris.iris_6_score_data import ScoreDataStep


@pytest.fixture(scope="function")
def fixture_score_data_step():
    """Fixture for ScoreDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = ScoreDataStep()
        yield sds


@patch("template_pipelines.steps.ml_iris.iris_6_score_data.len")
@patch("template_pipelines.steps.ml_iris.iris_6_score_data.json")
@patch("template_pipelines.steps.ml_iris.iris_6_score_data.pickle")
@patch("template_pipelines.steps.ml_iris.iris_6_score_data.open")
@patch("template_pipelines.steps.ml_iris.iris_6_score_data.pd")
def test_train_model_step_run(
    mock_pd, mock_open, mock_pickle, mock_json, mock_len, fixture_score_data_step
):
    """Test run."""
    mock_len.return_value = 1

    fixture_score_data_step.run()

    assert fixture_score_data_step.ioctx.get_fn.call_count == 3
    mock_open.assert_called()
    mock_pickle.load.assert_called()
    mock_json.load.assert_called()
    mock_json.dump.assert_called()
    mock_pd.read_parquet.assert_called_once()
    fixture_score_data_step.logger.info.assert_called()
