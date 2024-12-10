"""Unit tests for opinionated_pipelines/steps/iris_2_standarize_data.py."""

from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.ml_observability.iris_2_standarize_data import StandardizeDataStep


@pytest.fixture(scope="function")
def fixture_standardize_data_step():
    """Fixture for StandardizeDataStep step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        sds = StandardizeDataStep()
        yield sds


@patch("template_pipelines.steps.ml_observability.iris_2_standarize_data.pd")
def test_standardize_data_step_run(mock_pd, fixture_standardize_data_step):
    """Test run()."""
    fixture_standardize_data_step.ioctx.get_fn.return_value = "/"

    mock_data = MagicMock()
    mock_data.columns = ["COL1", "COL2"]
    mock_pd.read_parquet.return_value = mock_data

    fixture_standardize_data_step.run()

    mock_pd.read_parquet.assert_called()
    fixture_standardize_data_step.ioctx.get_fn.assert_called_once_with("uploaded_data.parquet")
    fixture_standardize_data_step.ioctx.get_output_fn.assert_called_once_with("standarised.csv")
    fixture_standardize_data_step.logger.info.assert_called()
    assert mock_data.columns == ["col1", "col2"]
