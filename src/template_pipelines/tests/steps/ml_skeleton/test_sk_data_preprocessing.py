"""Unit tests for sk_data_preprocessing.py."""

from unittest.mock import Mock

import pytest

from aif.pyrogai.tests.conftest import mock_step_env
from template_pipelines.steps.ml_skeleton.sk_data_preprocessing import Preprocessing


# This is other example of testing step from your pipeline. It uses mock from pyrogai conftest file.
@pytest.fixture
def fixture_preprocessing(request):
    """Fixture for step."""
    with mock_step_env(request):
        yield Preprocessing()


@pytest.mark.parametrize(
    "fixture_preprocessing",
    (
        {
            "platform": "Local",
            "config_module": "template_pipelines.config",
            "pipeline_name": "ml_skeleton",
        },
    ),
    indirect=True,
)
def test_step_preprocessing_run(fixture_preprocessing):
    """Test run()."""
    fixture_preprocessing.logger = Mock()

    fixture_preprocessing.run()

    assert fixture_preprocessing.logger.info.call_count == 2
