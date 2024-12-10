"""Tests for the validate_crunch_outputs step."""
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from template_pipelines.steps.crunch_tutorial.validate_crunch_outputs import ValidateCrunchOutputs


@pytest.fixture(scope="function")
def fixture_validate_crunch_outputs_step():
    """Step fixture."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        validate_crunch_outputs_step = ValidateCrunchOutputs()
        validate_crunch_outputs_step.inputs = {"sales_data_output": ""}
        yield validate_crunch_outputs_step


def test_validate_crunch_outputs_step_run(fixture_validate_crunch_outputs_step):
    """Test run."""
    with does_not_raise():
        fixture_validate_crunch_outputs_step.run()
