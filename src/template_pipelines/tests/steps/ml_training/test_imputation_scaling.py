"""Unit tests for opinionated_pipelines/steps/imputation_scaling.py."""
from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.ml_training.imputation_scaling import ImputationScaling


@pytest.fixture(scope="function")
def fixture_imputation_scaling():
    """Fixture for ImputationScaling step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        imputation_scaling = ImputationScaling()
        yield imputation_scaling


@patch.object(ImputationScaling, "split_data_into_unredeemed_and_redeemed")
@patch.object(ImputationScaling, "transform_and_save")
@patch.object(ImputationScaling, "get_preprocessor_pipeline")
@patch("template_pipelines.steps.ml_training.imputation_scaling.train_test_split")
@patch("template_pipelines.steps.ml_training.imputation_scaling.pd")
def test_imputation_scaling_run(
    mock_pd,
    mock_train_test_split,
    mock_get_pipeline,
    mock_transform_and_save,
    mock_split_data_into_unredeemed_and_redeemed,
    fixture_imputation_scaling,
):
    """Test run."""
    fixture_imputation_scaling.config = {
        "ml_training": {"random_state": 123, "target": "redemption_status"}
    }
    mock_train_test_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_transform_and_save.return_value = (MagicMock(), MagicMock())

    fixture_imputation_scaling.run()

    mock_train_test_split.assert_called()
    mock_get_pipeline.assert_called_once()
    mock_transform_and_save.assert_called_once()
    mock_split_data_into_unredeemed_and_redeemed.assert_called_once()
    fixture_imputation_scaling.logger.info.assert_called()
