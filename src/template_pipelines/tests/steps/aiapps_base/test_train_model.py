"""Unit tests for template_pipelines/steps/aiapps_base/train_model.py."""
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.aiapps_base.train_model import LogisticRegression, TrainModelStep


@pytest.fixture(scope="function")
def fixture_train_model_step():
    """Fixture for TrainModelStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        tms = TrainModelStep()
        yield tms


def test_train_model_step_train_model(mocker, fixture_train_model_step):
    """Test train_model()."""
    mocked_x = np.array([[1, 2], [3, 4]])
    mocked_y = np.array([0, 1])
    logistic_regression_mock = Mock(spec=LogisticRegression)
    logistic_regression_mock.fit.return_value = MagicMock()
    mocker.patch(
        "template_pipelines.steps.aiapps_base.train_model.LogisticRegression",
        return_value=logistic_regression_mock,
    )

    fixture_train_model_step.train_model(mocked_x, mocked_y)

    logistic_regression_mock.fit.assert_called_once_with(mocked_x, mocked_y)


@patch("template_pipelines.steps.aiapps_base.train_model.pd")
@patch("template_pipelines.steps.aiapps_base.train_model.pickle")
@patch("template_pipelines.steps.aiapps_base.train_model.open")
def test_train_model_step_run(mock_open, mock_pickle, mock_pd, fixture_train_model_step):
    """Test run."""
    mocked_data = {"col_1": [1, 3, 5], "col_2": [2, 4, 6], "target": [0, 1, 0]}
    mock_df = pd.DataFrame(mocked_data)

    mock_pd.read_csv.return_value = mock_df

    with patch.object(fixture_train_model_step, "train_model") as mock_train_model:
        fixture_train_model_step.run()

    mock_pd.read_csv.assert_called_once()
    mock_train_model.assert_called_once()
    fixture_train_model_step.ioctx.get_fn.assert_called_once()
    fixture_train_model_step.ioctx.get_output_fn.assert_called_once()
    mock_pickle.dump.assert_called()
    mock_open.assert_called()
