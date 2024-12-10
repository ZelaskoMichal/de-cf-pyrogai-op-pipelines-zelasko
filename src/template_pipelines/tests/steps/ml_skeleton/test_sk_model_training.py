"""Unit tests for sk_model_training.py."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.ml_skeleton.sk_model_training import ModelTraining


@pytest.fixture(scope="function")
def fixture_model_training():
    """Fixture for ModelTraining step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        mt = ModelTraining()
        yield mt


def test_feature_creation_run(fixture_model_training):
    """Test run()."""
    fixture_model_training.run()

    assert fixture_model_training.logger.info.call_count == 2
