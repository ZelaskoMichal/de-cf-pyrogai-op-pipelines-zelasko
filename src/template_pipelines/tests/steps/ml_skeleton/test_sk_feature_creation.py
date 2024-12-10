"""Unit tests for sk_feature_creation.py."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.ml_skeleton.sk_feature_creation import FeatureCreation


@pytest.fixture(scope="function")
def fixture_feature_creation():
    """Fixture for feature_creation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        fc = FeatureCreation()
        yield fc


def test_feature_creation_run(fixture_feature_creation):
    """Test run()."""
    fixture_feature_creation.run()

    assert fixture_feature_creation.logger.info.call_count == 2
