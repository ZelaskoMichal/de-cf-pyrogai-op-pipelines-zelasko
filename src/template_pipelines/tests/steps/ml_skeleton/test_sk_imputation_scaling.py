"""Unit tests for sk_imputation_scaling.py."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.ml_skeleton.sk_imputation_scaling import ImputationScaling


@pytest.fixture(scope="function")
def fixture_imputation_scaling():
    """Fixture for ImputationScaling step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        imputation = ImputationScaling()
        yield imputation


def test_feature_creation_run(fixture_imputation_scaling):
    """Test run()."""
    fixture_imputation_scaling.run()

    assert fixture_imputation_scaling.logger.info.call_count == 2
