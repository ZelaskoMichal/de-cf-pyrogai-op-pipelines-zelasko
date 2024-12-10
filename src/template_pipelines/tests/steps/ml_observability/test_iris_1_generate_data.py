"""Unit tests for opinionated_pipelines/steps/iris_1_generate_data.py."""

from unittest.mock import Mock, create_autospec, patch

import pandas as pd
import pytest

from template_pipelines.steps.ml_observability.iris_1_generate_data import GenerateDataStep


@pytest.fixture(scope="function")
def fixture_generate_data_step():
    """Fixture for GenerateDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        gds = GenerateDataStep()
        yield gds


def test_generate_data_step_generate_dataframe(fixture_generate_data_step):
    """Test generate_dataframe()."""
    df = fixture_generate_data_step.generate_dataframe()

    assert df.shape[1] == 5
    assert isinstance(df, pd.DataFrame)


@patch("template_pipelines.steps.ml_observability.datasets", create=True)
def test_generate_data_step_run(dataset_patch, fixture_generate_data_step):
    """Test run()."""
    mock_df = create_autospec(pd.DataFrame)
    fixture_generate_data_step.generate_dataframe = Mock(return_value=mock_df)
    fixture_generate_data_step.ioctx.get_output_fn.return_value = "/"

    fixture_generate_data_step.run()

    fixture_generate_data_step.ioctx.get_output_fn.assert_called_once_with("uploaded_data.parquet")
    mock_df.to_parquet.assert_called_once_with("/")
