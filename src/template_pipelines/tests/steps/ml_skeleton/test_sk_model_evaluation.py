"""Unit tests for sk_model_evaluation.py."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.ml_skeleton.sk_model_evaluation import ModelEvaluation


@pytest.fixture(scope="function")
def fixture_model_evaluation():
    """Fixture for ModelEvaluation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        me = ModelEvaluation()
        yield me


def test_feature_creation_run(fixture_model_evaluation):
    """Test run()."""
    fixture_model_evaluation.run()

    assert fixture_model_evaluation.logger.info.call_count == 2
