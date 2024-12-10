"""Unit tests for template_pipelines/steps/aiapps_base/preprocess_data.py."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.aiapps_base.preprocess_data import PreprocessDataStep, StandardScaler


@pytest.fixture(scope="function")
def fixture_preprocess_data_step():
    """Fixture for PreprocessDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = PreprocessDataStep()
        yield sds


def test_preprocess_data_step_preprocess_data(mocker, fixture_preprocess_data_step):
    """Test preprocess_data()."""
    mocked_x = np.array([[1, 2], [3, 4], [5, 6]])

    scaler_mock = Mock(spec=StandardScaler)
    scaler_mock.fit_transform.return_value = mocked_x
    mocker.patch(
        "template_pipelines.steps.aiapps_base.preprocess_data.StandardScaler",
        return_value=scaler_mock,
    )

    x_scaled = fixture_preprocess_data_step.preprocess_data(mocked_x)

    assert isinstance(x_scaled, np.ndarray)
    assert x_scaled.shape == mocked_x.shape


@patch("template_pipelines.steps.aiapps_base.preprocess_data.pd")
def test_standardize_data_step_run(mock_pd, fixture_preprocess_data_step):
    """Test run()."""
    fixture_preprocess_data_step.ioctx.get_fn.return_value = "/"

    mocked_data = {"col_1": [1, 3, 5], "col_2": [2, 4, 6], "target": [0, 1, 0]}
    mocked_df = pd.DataFrame(mocked_data)
    mock_pd.read_pickle.return_value = mocked_df

    with patch.object(fixture_preprocess_data_step, "preprocess_data") as mock_preprocess_data:
        fixture_preprocess_data_step.run()

    mock_pd.read_pickle.assert_called_once()
    mock_preprocess_data.assert_called_once()

    fixture_preprocess_data_step.ioctx.get_fn.assert_called_once_with("train_data.pkl")
    fixture_preprocess_data_step.ioctx.get_output_fn.assert_called_once_with(
        "preprocessed_data.csv"
    )

    fixture_preprocess_data_step.logger.info.assert_called()
