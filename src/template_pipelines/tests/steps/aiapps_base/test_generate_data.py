"""Unit tests for template_pipelines/steps/aiapps_base/generate_data.py."""

from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.aiapps_base.generate_data import GenerateDataStep


@pytest.fixture(scope="function")
def fixture_generate_data_step():
    """Fixture for GenerateDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        gds = GenerateDataStep()
        yield gds


def test_generate_data_step_generate_data(mocker, fixture_generate_data_step):
    """Test generate_data()."""
    mocked_x = np.array([[1, 2], [3, 4], [5, 6]])
    mocked_y = [0, 1, 0]

    make_classification_mock = mocker.patch(
        "template_pipelines.steps.aiapps_base.generate_data.make_classification",
        return_value=(mocked_x, mocked_y),
    )

    df = fixture_generate_data_step.generate_data(samples=100, features=5, informative=2)

    make_classification_mock.assert_called_once()

    assert df.shape == (3, 3)
    assert df.values.tolist() == [[1, 2, 0], [3, 4, 1], [5, 6, 0]]

    assert isinstance(df, pd.DataFrame)


def test_generate_data_step_run(fixture_generate_data_step):
    """Test run()."""
    mock_df = create_autospec(pd.DataFrame)
    fixture_generate_data_step.generate_data = Mock(return_value=mock_df)
    fixture_generate_data_step.ioctx.get_output_fn.return_value = "/"

    fixture_generate_data_step.run()

    fixture_generate_data_step.ioctx.get_output_fn.assert_called_once_with("train_data.pkl")
    mock_df.to_pickle.assert_called_once_with("/")
