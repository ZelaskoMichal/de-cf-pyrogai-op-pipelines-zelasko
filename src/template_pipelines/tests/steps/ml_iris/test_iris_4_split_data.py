"""Unit tests for template_pipelines/steps/iris_4_split_data.py."""

from unittest.mock import patch

import pandas as pd
import pytest

from template_pipelines.steps.ml_iris.iris_4_split_data import SplitDataStep


@pytest.fixture(scope="function")
def fixture_split_data_step():
    """Fixture for SplitDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = SplitDataStep()
        yield sds


def test_split_data_step_custom_train_test_split(fixture_split_data_step):
    """Test custom_train_test_split."""
    df1, df2 = fixture_split_data_step.custom_train_test_split()

    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)


@patch("template_pipelines.steps.ml_iris.iris_4_split_data.pd")
def test_split_data_step_run(mock_pd, fixture_split_data_step):
    """Test run."""
    fixture_split_data_step.run()

    fixture_split_data_step.ioctx.get_fn.assert_called_with("fixed.csv")
    mock_pd.read_csv.assert_called_once()
    assert fixture_split_data_step.ioctx.get_output_fn.call_count == 2
    fixture_split_data_step.logger.info.assert_called()
